"""
app.py — Streamlit UI for the Multi-Agent Tech Support Network
==============================================================

This file builds the interactive web interface for the support system.
It uses the same LangGraph pipeline from notebook.ipynb,
but wraps it in a clean chat interface with live agent trace visibility.

HOW TO RUN:
    streamlit run app.py

WHAT YOU'LL SEE:
    - A chat box where you type your tech question
    - The final answer displayed clearly
    - An expandable "Agent Trace" panel showing every decision made
"""

import os
import json
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, List

# Load environment variables from .env file
load_dotenv()

# ================================================================
# PAGE CONFIGURATION
# Must be the FIRST Streamlit command in the file
# ================================================================
st.set_page_config(
    page_title="🤖 AI Tech Support Hub",
    page_icon="🤖",
    layout="wide",          # Use full browser width
    initial_sidebar_state="expanded"
)

# ================================================================
# IMPORTS — LangChain / LangGraph components
# ================================================================
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# ================================================================
# STATE DEFINITION
# The "shared notebook" that all agents read and write to
# ================================================================
class GraphState(TypedDict):
    original_query: str       # User's raw question
    optimized_query: str      # AI-rewritten search query
    documents: List[Document] # Retrieved docs (from FAISS or web)
    generation: str           # Generated answer text
    loop_count: int           # Number of correction retries
    web_search_used: bool     # Whether web search was triggered
    all_relevant: bool        # Whether local docs passed relevance check
    agent_trace: List[str]    # Step-by-step log of agent decisions


# ================================================================
# VECTOR STORE SETUP (cached so it only runs once)
# @st.cache_resource tells Streamlit: "only build this once,
# then reuse it for all users and reruns"
# ================================================================
@st.cache_resource
def build_vector_store():
    """
    Loads sample technical documentation, splits into chunks,
    converts to embeddings, and stores in FAISS for fast search.
    
    This only runs once when the app starts — Streamlit caches it.
    """
    # Sample tech documentation (replace with your actual files!)
    sample_docs = [
        Document(page_content="""
How to Reset Your Password:
1. Navigate to the login page at app.example.com
2. Click 'Forgot Password' below the login button
3. Enter your registered email address
4. Check your email for a reset link (valid for 24 hours)
5. Click the link and enter a new password (min 8 chars, 1 uppercase, 1 number)
6. Log in with your new password
Note: If no email arrives within 5 minutes, check your spam folder.
        """, metadata={"source": "user_manual.txt"}),

        Document(page_content="""
API Rate Limiting Policy:
Our API enforces rate limits to ensure fair usage:
- Free Tier: 100 requests/minute, 10,000 requests/day
- Pro Tier: 1,000 requests/minute, 500,000 requests/day
- Enterprise: Unlimited (contact sales)
When the rate limit is exceeded, the API returns HTTP 429 (Too Many Requests).
Implement exponential backoff: wait 1s, then 2s, then 4s between retries.
Response headers include X-RateLimit-Remaining to track remaining quota.
        """, metadata={"source": "api_docs.txt"}),

        Document(page_content="""
Installation Guide for Windows:
Requirements: Windows 10/11, 4GB RAM minimum, 2GB disk space
Steps:
1. Download the installer from downloads.example.com
2. Right-click the .exe file and select 'Run as Administrator'
3. Accept the license agreement
4. Choose installation directory (default: C:\\Program Files\\ExampleApp)
5. Click Install and wait for completion (~3 minutes)
6. Launch from the Start Menu
Troubleshooting: If you see 'DLL not found', install Visual C++ Redistributable from Microsoft.
        """, metadata={"source": "install_guide.txt"}),

        Document(page_content="""
Error Code Reference:
ERR_001: Authentication failed - Check credentials or re-login
ERR_002: Connection timeout - Check internet or firewall settings
ERR_003: File not found - Verify the file path and permissions
ERR_004: Insufficient storage - Free up at least 500MB disk space
ERR_005: License expired - Contact support@example.com to renew
ERR_429: Rate limit exceeded - Implement backoff and retry logic
ERR_500: Server error - Our team is notified; retry in 5 minutes
        """, metadata={"source": "error_codes.txt"}),

        Document(page_content="""
Two-Factor Authentication (2FA) Setup:
Enabling 2FA adds an extra security layer to your account.
Supported methods: Authenticator App (recommended), SMS, Email OTP
Setup steps:
1. Go to Account Settings > Security
2. Click 'Enable Two-Factor Authentication'
3. Choose your preferred method
4. For Authenticator App: scan the QR code with Google Authenticator or Authy
5. Enter the 6-digit code to confirm
6. Save your backup codes in a secure location!
If you lose access to your 2FA device, use a backup code or contact support.
        """, metadata={"source": "security_guide.txt"}),
    ]

    # Split documents into smaller searchable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(sample_docs)

    # Create FAISS vector store from chunks
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return retriever


# ================================================================
# GRAPH BUILDER (also cached — only compiled once)
# ================================================================
@st.cache_resource
def build_graph(_retriever):
    """
    Builds and compiles the full LangGraph multi-agent pipeline.
    The underscore in _retriever tells Streamlit not to hash it
    (Streamlit can't hash complex objects like retrievers).
    """

    # Initialize LLM and tools
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tavily_tool = TavilySearchResults(max_results=3)

    # ----------------------------------------------------------
    # NODE 1: Query Rewriter (Support Agent)
    # Turns vague queries into precise search terms
    # ----------------------------------------------------------
    def rewrite_query_node(state: GraphState) -> dict:
        query = state["original_query"]
        trace = state.get("agent_trace", [])
        loop = state.get("loop_count", 0)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Search Query Optimizer.
Rewrite the user's tech support question into precise, keyword-driven search terms.
Output ONLY the optimized query — no explanation."""),
            ("human", "Rewrite this query: {query}")
        ])

        result = (prompt | llm).invoke({"query": query})
        optimized = result.content.strip()

        trace.append(f"🔄 **Query Rewriter** (attempt {loop + 1}): `{query}` → `{optimized}`")
        return {"optimized_query": optimized, "agent_trace": trace}

    # ----------------------------------------------------------
    # NODE 2: Document Retriever (Support Agent)
    # Searches FAISS for most relevant doc chunks
    # ----------------------------------------------------------
    def retrieve_docs_node(state: GraphState) -> dict:
        trace = state.get("agent_trace", [])
        docs = _retriever.invoke(state["optimized_query"])
        sources = list(set([d.metadata.get("source", "unknown") for d in docs]))
        trace.append(f"📚 **Retriever**: Found {len(docs)} chunks from local docs: {', '.join(sources)}")
        return {"documents": docs, "web_search_used": False, "agent_trace": trace}

    # ----------------------------------------------------------
    # NODE 3: Document Grader (QA Agent)
    # Judges whether retrieved docs are actually relevant
    # ----------------------------------------------------------
    def grade_documents_node(state: GraphState) -> dict:
        query = state["original_query"]
        documents = state["documents"]
        trace = state.get("agent_trace", [])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Document Relevance Grader.
Output ONLY valid JSON: {"relevance_score": "yes"} or {"relevance_score": "no"}
"yes" if the document helps answer the question. "no" if it's off-topic."""),
            ("human", "Question: {query}\n\nDocument:\n{document}")
        ])

        relevant_docs = []
        grades = []

        for doc in documents:
            try:
                result = (prompt | llm).invoke({"query": query, "document": doc.page_content})
                data = json.loads(result.content.strip())
                score = data.get("relevance_score", "no")
            except (json.JSONDecodeError, Exception):
                score = "no"

            if score == "yes":
                relevant_docs.append(doc)
            grades.append(score)

        all_relevant = len(relevant_docs) > 0
        grade_summary = f"{grades.count('yes')}/{len(grades)} docs relevant"

        if all_relevant:
            trace.append(f"✅ **QA Grader**: {grade_summary} → Proceeding to generate answer")
        else:
            trace.append(f"⚠️ **QA Grader**: {grade_summary} → Local docs insufficient, triggering web search")

        return {
            "documents": relevant_docs if all_relevant else documents,
            "all_relevant": all_relevant,
            "agent_trace": trace
        }

    # ----------------------------------------------------------
    # NODE 4: Web Search Fallback (Support Agent + Tavily)
    # Searches the live internet when local docs don't help
    # ----------------------------------------------------------
    def web_search_node(state: GraphState) -> dict:
        trace = state.get("agent_trace", [])

        try:
            results = tavily_tool.invoke({"query": state["optimized_query"]})
            web_docs = [
                Document(
                    page_content=r.get("content", ""),
                    metadata={"source": r.get("url", "web"), "type": "web"}
                )
                for r in results
            ]
            trace.append(f"🌐 **Web Search**: Retrieved {len(web_docs)} results from the internet")
        except Exception as e:
            # Graceful fallback if Tavily API fails
            web_docs = [Document(
                page_content=f"Web search unavailable: {str(e)}",
                metadata={"source": "error"}
            )]
            trace.append(f"❌ **Web Search**: Failed ({str(e)[:60]})")

        return {"documents": web_docs, "web_search_used": True, "agent_trace": trace}

    # ----------------------------------------------------------
    # NODE 5: Answer Generator (Support Agent)
    # Creates the actual technical support response
    # ----------------------------------------------------------
    def generate_answer_node(state: GraphState) -> dict:
        query = state["original_query"]
        documents = state["documents"]
        trace = state.get("agent_trace", [])

        context = "\n\n---\n\n".join([doc.page_content for doc in documents])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Tech Support Specialist.
Provide clear, accurate, step-by-step technical answers.

RULES:
1. ONLY use information from the provided context
2. Format with numbered steps when applicable
3. Be concise but thorough
4. If context is insufficient, say so clearly

Context:
{context}"""),
            ("human", "Question: {query}")
        ])

        result = (prompt | llm).invoke({"context": context, "query": query})
        trace.append(f"✍️ **Answer Generator**: Response created ({len(result.content)} chars)")

        return {"generation": result.content, "agent_trace": trace}

    # ----------------------------------------------------------
    # NODE 6: Hallucination Checker (QA Agent)
    # Verifies the answer against source documents
    # ----------------------------------------------------------
    def hallucination_check_node(state: GraphState) -> dict:
        generation = state["generation"]
        documents = state["documents"]
        loop_count = state.get("loop_count", 0)
        trace = state.get("agent_trace", [])

        context = "\n\n".join([doc.page_content for doc in documents])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Hallucination Detection Judge.
Verify if the AI answer is fully supported by the source documents.

Output ONLY valid JSON:
{"hallucination_check": "passed"} — every claim is sourced from documents
{"hallucination_check": "failed"} — answer contains invented or unsupported claims

Source Documents:
{context}"""),
            ("human", "AI Answer:\n{generation}")
        ])

        try:
            result = (prompt | llm).invoke({"context": context, "generation": generation})
            data = json.loads(result.content.strip())
            check = data.get("hallucination_check", "failed")
        except (json.JSONDecodeError, Exception):
            check = "failed"

        if check == "passed":
            trace.append("✅ **QA Hallucination Judge**: Answer is grounded and accurate → APPROVED")
            return {"loop_count": loop_count, "agent_trace": trace}
        else:
            new_count = loop_count + 1
            trace.append(f"❌ **QA Hallucination Judge**: Hallucination detected (retry {new_count}/2) → Sending back for rewrite")
            return {"loop_count": new_count, "agent_trace": trace}

    # ----------------------------------------------------------
    # ROUTING FUNCTIONS (Conditional Edges)
    # ----------------------------------------------------------
    def route_after_grading(state: GraphState) -> str:
        return "generate" if state.get("all_relevant", False) else "web_search"

    def route_after_hallucination_check(state: GraphState) -> str:
        last_trace = state.get("agent_trace", [""])[-1]
        if "APPROVED" in last_trace:
            return "end"
        elif state.get("loop_count", 0) >= 2:
            return "end"
        return "rewrite"

    # ----------------------------------------------------------
    # ASSEMBLE THE GRAPH
    # ----------------------------------------------------------
    workflow = StateGraph(GraphState)

    # Add all nodes
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("retrieve_docs", retrieve_docs_node)
    workflow.add_node("grade_docs", grade_documents_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_answer_node)
    workflow.add_node("hallucination_check", hallucination_check_node)

    # Set starting node
    workflow.set_entry_point("rewrite_query")

    # Add fixed edges (always go this direction)
    workflow.add_edge("rewrite_query", "retrieve_docs")
    workflow.add_edge("retrieve_docs", "grade_docs")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "hallucination_check")

    # Add conditional edges (branching)
    workflow.add_conditional_edges(
        "grade_docs",
        route_after_grading,
        {"generate": "generate", "web_search": "web_search"}
    )
    workflow.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination_check,
        {"end": END, "rewrite": "rewrite_query"}
    )

    return workflow.compile()


# ================================================================
# STREAMLIT UI
# ================================================================

# --- Custom CSS for a polished look ---
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    /* Answer box styling */
    .answer-box {
        background: #f0f7ff;
        border-left: 4px solid #0066cc;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: black;
    }
    /* Chat message styling */
    .user-msg {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #2196F3;
    }
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🤖 Adaptive Tech Support Hub</h1>
    <p>Powered by Multi-Agent AI • LangGraph + GPT-4o-mini</p>
    <p style="font-size: 0.85rem; opacity: 0.8;">Two AI agents collaborate to find accurate, grounded answers to your tech questions</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar: Architecture explainer ---
with st.sidebar:
    st.header("🏗️ How It Works")
    st.markdown("""
    **Two AI Agents collaborate:**
    
    🔧 **Support Specialist**
    - Rewrites your query for better search
    - Retrieves relevant docs from local database
    - Searches the web if local docs fall short
    - Generates the final answer
    
    🔍 **QA Engineer (Gatekeeper)**
    - Grades retrieved documents for relevance
    - Checks the answer for hallucinations
    - Triggers self-correction if quality fails
    
    ---
    
    **Flow:**
    ```
    Your Query
        ↓
    Rewrite Query
        ↓
    Retrieve Docs (FAISS)
        ↓
    Grade Docs (QA)
       ↓         ↓
    Generate  Web Search
       ↓         ↓
    Generate Answer
        ↓
    Hallucination Check (QA)
       ↓              ↓
      ✅ END      ❌ Retry (max 2x)
    ```
    
    ---
    
    **Setup Required:**
    Add to `.env` file:
    ```
    OPENAI_API_KEY=sk-...
    TAVILY_API_KEY=tvly-...
    ```
    """)

    st.divider()
    st.caption("Built with LangGraph + Streamlit")

# --- Check for API keys ---
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ **OPENAI_API_KEY not found.** Please add it to your `.env` file and restart the app.")
    st.stop()

if not os.getenv("TAVILY_API_KEY"):
    st.warning("⚠️ **TAVILY_API_KEY not found.** Web search fallback will not work, but local doc search will still function.")

# --- Initialize components (cached) ---
with st.spinner("⚙️ Initializing AI agents and vector store..."):
    retriever = build_vector_store()
    graph_app = build_graph(retriever)

# --- Chat History ---
# st.session_state persists data between Streamlit reruns
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Display past conversations ---
if st.session_state.chat_history:
    st.subheader("💬 Conversation History")

    for i, exchange in enumerate(st.session_state.chat_history):
        # User message
        with st.container():
            st.markdown(f"""
            <div class="user-msg">
                <strong>👤 You:</strong> {exchange['query']}
            </div>
            """, unsafe_allow_html=True)

        # Agent response
        with st.container():
            # Show badges for what happened
            cols = st.columns([1, 1, 1, 4])
            with cols[0]:
                if exchange.get("web_search_used"):
                    st.markdown("🌐 **Web Used**")
                else:
                    st.markdown("📚 **Local Docs**")
            with cols[1]:
                loops = exchange.get("loop_count", 0)
                if loops > 0:
                    st.markdown(f"🔁 **{loops} Retry(s)**")
                else:
                    st.markdown("✅ **1st Try**")

            # The actual answer
            st.markdown(f"""
            <div class="answer-box">
                <strong>🤖 Support Agent:</strong><br><br>{exchange['answer']}
            </div>
            """, unsafe_allow_html=True)

            # Expandable agent trace
            with st.expander(f"🔍 View Agent Decision Trace (Exchange {i+1})", expanded=False):
                st.markdown("**Step-by-step decisions made by both agents:**")
                for step_num, step in enumerate(exchange.get("trace", []), 1):
                    st.markdown(f"**{step_num}.** {step}")

        st.divider()

# --- Query Input ---
st.subheader("🎯 Ask a Tech Question")

# Example queries to help users get started
st.markdown("**Try these examples:**")
example_cols = st.columns(3)
with example_cols[0]:
    if st.button("🔑 Password reset help"):
        st.session_state.example_query = "how do i reset my password i forgot it"
with example_cols[1]:
    if st.button("⚡ API rate limit error"):
        st.session_state.example_query = "getting error 429 from the API what does it mean"
with example_cols[2]:
    if st.button("🔐 Set up 2FA"):
        st.session_state.example_query = "how to enable two factor authentication on my account"

# Text input (pre-filled if example was clicked)
default_text = st.session_state.get("example_query", "")
user_query = st.text_area(
    "Type your technical question here:",
    value=default_text,
    height=100,
    placeholder="e.g. 'I'm getting a connection error after the update, how do I fix it?'"
)

# Clear the example query after displaying it
if "example_query" in st.session_state:
    del st.session_state.example_query

submit_col, clear_col = st.columns([1, 5])
with submit_col:
    submit = st.button("🚀 Ask Agents", type="primary", use_container_width=True)
with clear_col:
    if st.button("🗑️ Clear History"):
        st.session_state.chat_history = []
        st.rerun()

# ================================================================
# MAIN EXECUTION — runs when user clicks "Ask Agents"
# ================================================================
if submit and user_query.strip():

    st.divider()
    st.subheader("⚙️ Agents Working...")

    # Status container for live updates
    with st.status("🤖 Multi-Agent Pipeline Running...", expanded=True) as status:

        # Show which node is currently running using live streaming
        st.write("📡 Streaming agent execution in real-time...")

        initial_state = {
            "original_query": user_query.strip(),
            "optimized_query": "",
            "documents": [],
            "generation": "",
            "loop_count": 0,
            "web_search_used": False,
            "all_relevant": False,
            "agent_trace": []
        }

        final_state = None
        node_order = []

        # Stream through each node's execution
        try:
            for step_output in graph_app.stream(initial_state, stream_mode="updates"):
                node_name = list(step_output.keys())[0]
                node_data = step_output[node_name]
                node_order.append(node_name)

                # Map node names to friendly labels
                node_labels = {
                    "rewrite_query": "🔄 Rewriting Query",
                    "retrieve_docs": "📚 Retrieving Documents",
                    "grade_docs": "🔍 Grading Document Relevance",
                    "web_search": "🌐 Searching the Web",
                    "generate": "✍️ Generating Answer",
                    "hallucination_check": "🧪 Checking for Hallucinations"
                }

                label = node_labels.get(node_name, node_name)
                st.write(f"✓ {label}")

                # Build final state incrementally
                if final_state is None:
                    final_state = {**initial_state, **node_data}
                else:
                    final_state.update(node_data)

            status.update(label="✅ All agents completed successfully!", state="complete")

        except Exception as e:
            status.update(label=f"❌ Error: {str(e)}", state="error")
            st.error(f"An error occurred: {str(e)}")
            st.stop()

    # ============================================================
    # DISPLAY RESULTS
    # ============================================================
    if final_state and final_state.get("generation"):

        st.divider()
        st.subheader("💬 Answer")

        # --- Info badges ---
        badge_cols = st.columns(4)
        with badge_cols[0]:
            st.metric("🔁 Retry Loops", final_state.get("loop_count", 0))
        with badge_cols[1]:
            source = "🌐 Web + Local" if final_state.get("web_search_used") else "📚 Local Docs"
            st.metric("📂 Data Source", source)
        with badge_cols[2]:
            doc_count = len(final_state.get("documents", []))
            st.metric("📄 Docs Used", doc_count)
        with badge_cols[3]:
            nodes_run = len(set(node_order))
            st.metric("⚙️ Nodes Run", nodes_run)

        # --- The actual answer ---
        st.markdown(f"""
        <div class="answer-box">
            <strong>🤖 Tech Support Agent says:</strong><br><br>
            {final_state['generation'].replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)

        # --- Agent Trace Panel ---
        with st.expander("🔍 View Full Agent Decision Trace", expanded=True):
            st.markdown("**Detailed step-by-step log of every decision made:**")
            st.markdown("")

            trace = final_state.get("agent_trace", [])
            if trace:
                for step_num, step in enumerate(trace, 1):
                    # Color-code based on content
                    if "❌" in step:
                        st.error(f"**Step {step_num}:** {step}")
                    elif "⚠️" in step:
                        st.warning(f"**Step {step_num}:** {step}")
                    elif "✅" in step or "🌐" in step:
                        st.success(f"**Step {step_num}:** {step}")
                    else:
                        st.info(f"**Step {step_num}:** {step}")
            else:
                st.write("No trace available.")

        # --- Save to history ---
        st.session_state.chat_history.append({
            "query": user_query.strip(),
            "answer": final_state["generation"],
            "trace": final_state.get("agent_trace", []),
            "web_search_used": final_state.get("web_search_used", False),
            "loop_count": final_state.get("loop_count", 0)
        })

elif submit and not user_query.strip():
    st.warning("⚠️ Please enter a question before clicking 'Ask Agents'.")

# --- Footer ---
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.85rem;">
    🤖 Multi-Agent AI Tech Support • Built with LangGraph + Streamlit<br>
    <em>Support Specialist Agent + QA Engineer Agent working together</em>
</div>
""", unsafe_allow_html=True)
