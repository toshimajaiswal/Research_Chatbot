import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.llm import get_chatgroq_model
from utils.rag_pipeline import add_document_to_db, retrieve_relevant_chunks, get_indexed_files
from utils.web_search import web_search
from utils.prompt import build_system_prompt
from utils.chat_history import (
    save_session, load_all_sessions,
    load_session_messages, delete_session, delete_all_sessions
)


def apply_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;900&family=Share+Tech+Mono&family=Exo+2:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Exo 2', sans-serif;
    }

    .stApp {
        background-color: #010408;
        background-image:
            linear-gradient(rgba(0, 255, 170, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 170, 0.03) 1px, transparent 1px);
        background-size: 40px 40px;
        color: #c8d6e5;
    }

    header[data-testid="stHeader"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    [data-testid="stToolbar"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    button[kind="header"],
    .stAppHeader,
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        position: fixed !important;
        top: -9999px !important;
    }

    [data-testid="stSidebar"] > div > div:first-child > button,
    [data-testid="stSidebar"] > div:first-child > button,
    section[data-testid="stSidebar"] button[kind="header"] {
        display: none !important;
    }

    [data-testid="stSidebar"] {
        background: #010408 !important;
        border-right: 1px solid rgba(0, 255, 170, 0.15);
        box-shadow: 4px 0 30px rgba(0, 255, 170, 0.05);
        min-width: 300px !important;
        max-width: 320px !important;
    }

    [data-testid="stSidebar"] * {
        font-family: 'Exo 2', sans-serif;
    }

    [data-testid="stSidebar"] h1 {
        font-family: 'Orbitron', monospace !important;
        font-size: 1.2rem !important;
        background: linear-gradient(90deg, #00ffaa, #00bfff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    h1 {
        font-family: 'Orbitron', monospace !important;
        background: linear-gradient(90deg, #00ffaa 0%, #00bfff 50%, #00ffaa 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s linear infinite;
        font-weight: 900 !important;
        letter-spacing: 3px !important;
        text-transform: uppercase;
    }

    @keyframes shimmer {
        0%   { background-position: 0% center; }
        100% { background-position: 200% center; }
    }

    h2, h3 {
        font-family: 'Orbitron', monospace !important;
        color: #00ffaa !important;
        font-size: 0.85rem !important;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    [data-testid="stChatMessage"] {
        background: rgba(0, 8, 20, 0.9) !important;
        border: 1px solid rgba(0, 191, 255, 0.12);
        border-radius: 4px !important;
        padding: 16px !important;
        margin-bottom: 10px !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }

    [data-testid="stChatMessage"]:hover {
        border-color: rgba(0, 255, 170, 0.3);
        box-shadow: 0 0 20px rgba(0, 255, 170, 0.06);
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        border-left: 2px solid #00bfff;
        background: rgba(0, 191, 255, 0.04) !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        border-left: 2px solid #00ffaa;
        background: rgba(0, 255, 170, 0.03) !important;
    }

    [data-testid="stChatInput"] {
        background: rgba(0, 8, 20, 0.95) !important;
        border: 1px solid rgba(0, 255, 170, 0.25) !important;
        border-radius: 4px !important;
        color: #00ffaa !important;
        font-family: 'Share Tech Mono', monospace !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: #00ffaa !important;
        box-shadow:
            0 0 0 1px rgba(0, 255, 170, 0.3),
            0 0 20px rgba(0, 255, 170, 0.1),
            0 0 40px rgba(0, 255, 170, 0.05) !important;
    }

    .stButton > button {
        background: transparent !important;
        border: 1px solid rgba(0, 255, 170, 0.3) !important;
        border-radius: 3px !important;
        color: #00ffaa !important;
        font-family: 'Orbitron', monospace !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        transition: all 0.2s ease !important;
        position: relative;
        overflow: hidden;
    }

    .stButton > button:hover {
        background: rgba(0, 255, 170, 0.08) !important;
        border-color: #00ffaa !important;
        box-shadow: 0 0 10px rgba(0, 255, 170, 0.2),
                    0 0 30px rgba(0, 255, 170, 0.08) !important;
        transform: translateY(-1px);
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        font-size: 0.68rem !important;
        text-align: left !important;
        letter-spacing: 0.5px !important;
        border-color: rgba(0, 255, 170, 0.08) !important;
        color: #7ec8e3 !important;
        padding: 6px 10px !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: rgba(0, 255, 170, 0.3) !important;
        color: #00ffaa !important;
        background: rgba(0, 255, 170, 0.05) !important;
    }

    [data-testid="stFileUploader"] {
        background: rgba(0, 8, 20, 0.8) !important;
        border: 1px dashed rgba(0, 255, 170, 0.25) !important;
        border-radius: 4px !important;
    }

    [data-testid="stExpander"] {
        background: rgba(0, 8, 20, 0.8) !important;
        border: 1px solid rgba(0, 191, 255, 0.15) !important;
        border-radius: 4px !important;
    }

    [data-testid="stExpander"]:hover {
        border-color: rgba(0, 191, 255, 0.3) !important;
    }

    [data-testid="stRadio"] label {
        color: #7ec8e3 !important;
        font-family: 'Exo 2', sans-serif !important;
        font-size: 0.9rem !important;
    }

    [data-testid="stToggle"] label {
        color: #7ec8e3 !important;
        font-family: 'Exo 2', sans-serif !important;
    }

    hr {
        border: none !important;
        border-top: 1px solid rgba(0, 255, 170, 0.1) !important;
        box-shadow: 0 1px 8px rgba(0, 255, 170, 0.05);
    }

    code {
        font-family: 'Share Tech Mono', monospace !important;
        background: rgba(0, 255, 170, 0.08) !important;
        color: #00ffaa !important;
        border: 1px solid rgba(0, 255, 170, 0.2);
        border-radius: 2px;
        padding: 2px 6px;
    }

    [data-testid="stCaptionContainer"] p {
        color: #2a5a6a !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.75rem !important;
        letter-spacing: 1px;
    }

    ::-webkit-scrollbar       { width: 3px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(0, 255, 170, 0.2); border-radius: 2px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(0, 255, 170, 0.5); }

    input, textarea { caret-color: #00ffaa !important; }
    ::selection     { background: rgba(0, 255, 170, 0.2); color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)


def get_chat_response(chat_model, messages, system_prompt):
    """Core function from base template — unchanged."""
    try:
        formatted_messages = [SystemMessage(content=system_prompt)]
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))
        return chat_model.invoke(formatted_messages).content
    except Exception as e:
        return f"Error: {str(e)}"


def _render_sources(sources: dict):
    if sources.get("rag"):
        st.markdown(
            "<p style='font-family:Share Tech Mono,monospace;color:#00ffaa;"
            "font-size:0.75rem;letter-spacing:2px;text-transform:uppercase;"
            "margin-bottom:6px;'>◈ From Uploaded Papers</p>",
            unsafe_allow_html=True
        )
        for c in sources["rag"]:
            st.markdown(
                f"""<div style="background:rgba(0,255,170,0.05);
                border:1px solid rgba(0,255,170,0.2);
                border-left:2px solid #00ffaa;border-radius:3px;
                padding:6px 12px;margin:4px 0;
                font-family:'Share Tech Mono',monospace;
                font-size:0.78rem;color:#00ffaa;">
                📄 {c['source']}
                <span style="color:#1a5a4a;margin-left:12px;">
                    relevance: {c['relevance']}
                </span></div>""",
                unsafe_allow_html=True
            )

    if sources.get("web"):
        st.markdown(
            "<p style='font-family:Share Tech Mono,monospace;color:#00bfff;"
            "font-size:0.75rem;letter-spacing:2px;text-transform:uppercase;"
            "margin:10px 0 6px 0;'>◈ From Live Web Search</p>",
            unsafe_allow_html=True
        )
        for r in sources["web"]:
            if r["url"]:
                st.markdown(
                    f"""<div style="background:rgba(0,191,255,0.04);
                    border:1px solid rgba(0,191,255,0.15);
                    border-left:2px solid #00bfff;border-radius:3px;
                    padding:6px 12px;margin:4px 0;font-size:0.82rem;">
                    🔗 <a href="{r['url']}" target="_blank"
                    style="color:#00bfff;text-decoration:none;
                    font-family:'Exo 2',sans-serif;font-weight:500;">
                    {r['title']}</a></div>""",
                    unsafe_allow_html=True
                )


def instructions_page():
    st.title("🧠 ResearchMind — Setup Guide")
    st.markdown("""
## What is ResearchMind?
An AI research assistant that:
- Upload multiple research PDFs and chat across all of them
- Pull live web results 
- Switch between Concise and Detailed response modes
- See exactly which source answered your question


## How to Use
1. Go to the **Chat** page
2. Upload research PDFs in the sidebar
3. Toggle **Live Web Search** on/off
4. Pick **Concise** or **Detailed** mode
5. Ask your research questions!

---
Ready? → Navigate to **Chat** in the sidebar!
""")


def chat_page():

    # ── Session state init — MUST be before any st.session_state access ──
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "active_session_id" not in st.session_state:
        st.session_state["active_session_id"] = None

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        mode = st.radio(
            "Response Mode",
            ["concise", "detailed"],
            format_func=lambda x: "Concise" if x == "concise" else "Detailed",
        )

        use_web = st.toggle("🌐 Live Web Search", value=True)

        # ── New Chat / Save Chat — always visible, never need to scroll ───
        st.divider()
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("New Chat", use_container_width=True):
                if st.session_state.messages and st.session_state["active_session_id"] is None:
                    save_session(st.session_state.messages)
                st.session_state.messages             = []
                st.session_state["active_session_id"] = None
                st.rerun()
        with col2:
            if st.button("Save Chat", use_container_width=True):
                if st.session_state.messages:
                    sid = save_session(
                        st.session_state.messages,
                        session_id=st.session_state["active_session_id"]
                    )
                    st.session_state["active_session_id"] = sid
                    st.toast("✅ Chat saved!")
                else:
                    st.toast("Nothing to save yet.")

        st.divider()
        st.header("📄 Upload Documents")

        uploaded_files = st.file_uploader(
            "Drop PDFs here", type=["pdf"], accept_multiple_files=True
        )

        if uploaded_files:
            for file in uploaded_files:
                key = f"indexed_{file.name}"
                if key not in st.session_state:
                    with st.spinner(f"Indexing {file.name}…"):
                        try:
                            n = add_document_to_db(file.name, file)
                            st.session_state[key] = True
                            st.success(f"✅ {file.name} — {n} chunks indexed")
                        except Exception as e:
                            st.error(f"❌ {file.name}: {e}")

        indexed = get_indexed_files()
        if indexed:
            st.divider()
            st.markdown("**Indexed Papers:**")
            for name in indexed:
                st.markdown(
                    f"""<div style="background:rgba(0,255,170,0.05);
                    border:1px solid rgba(0,255,170,0.2);
                    border-left:2px solid #00ffaa;border-radius:3px;
                    padding:6px 10px;margin:3px 0;font-size:0.78rem;
                    color:#00ffaa;font-family:'Share Tech Mono',monospace;">
                    📄 {name}</div>""",
                    unsafe_allow_html=True
                )

        st.divider()
        st.header("🕘 Chat History")

        sessions = load_all_sessions()
        if sessions:
            for s in sessions:
                col1, col2 = st.columns([5, 1])
                with col1:
                    if st.button(
                        f"💬 {s['title']}\n_{s['timestamp']}_",
                        key=f"load_{s['id']}",
                        use_container_width=True
                    ):
                        st.session_state.messages             = load_session_messages(s["id"])
                        st.session_state["active_session_id"] = s["id"]
                        st.rerun()
                with col2:
                    if st.button("🗑", key=f"del_{s['id']}"):
                        delete_session(s["id"])
                        st.rerun()

            st.divider()
            if st.button("Clear All History", use_container_width=True):
                delete_all_sessions()
                st.rerun()
        else:
            st.caption("No saved chats yet.")

    # ── Hero Header ────────────────────────────────────────────────────────
    st.markdown("""
        <div style="text-align:center;padding:2.5rem 0 1rem 0;">
            <div style="font-family:'Orbitron',monospace;font-size:0.75rem;
                letter-spacing:6px;color:#00ffaa;text-transform:uppercase;
                margin-bottom:1rem;opacity:0.7;">
                ◈ AI RESEARCH ASSISTANT ◈
            </div>
            <h1 style="margin:0;font-size:3.2rem;letter-spacing:6px;">
                RESEARCHMIND
            </h1>
            <p style="color:#2a5a6a;font-size:0.85rem;margin-top:0.8rem;
                font-family:'Share Tech Mono',monospace;letter-spacing:2px;">
                UPLOAD · RETRIEVE · SYNTHESIZE · CITE
            </p>
        </div>
        <div style="height:1px;
            background:linear-gradient(90deg,transparent,#00ffaa,#00bfff,transparent);
            margin-bottom:1.5rem;box-shadow:0 0 10px rgba(0,255,170,0.3);">
        </div>
    """, unsafe_allow_html=True)

    # ── Render conversation ────────────────────────────────────────────────
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    _render_sources(message["sources"])

    # ── Empty state ────────────────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("""
            <div style="text-align:center;padding:5rem 2rem;
                font-family:'Share Tech Mono',monospace;">
                <div style="font-size:2.5rem;margin-bottom:1.5rem;
                    filter:drop-shadow(0 0 12px rgba(0,255,170,0.4));">◈</div>
                <p style="font-size:0.9rem;color:#00ffaa;letter-spacing:3px;
                    text-transform:uppercase;margin-bottom:0.5rem;">
                    System Ready
                </p>
                <p style="font-size:0.75rem;color:#1a3a4a;letter-spacing:2px;
                    text-transform:uppercase;">
                    Upload a paper or ask a question to initialize
                </p>
            </div>
        """, unsafe_allow_html=True)

    # ── Chat input ─────────────────────────────────────────────────────────
    if prompt := st.chat_input("Hi! Ask anything about your research…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Researching…"):
                rag_chunks    = retrieve_relevant_chunks(prompt)
                web_results   = web_search(prompt) if use_web else []
                system_prompt = build_system_prompt(mode, rag_chunks, web_results)
                chat_model    = get_chatgroq_model(mode)
                response      = get_chat_response(
                    chat_model, st.session_state.messages, system_prompt
                )

            st.markdown(response)

            sources = {"rag": rag_chunks, "web": web_results}
            if rag_chunks or web_results:
                with st.expander("📚 Sources Used"):
                    _render_sources(sources)

        st.session_state.messages.append({
            "role":    "assistant",
            "content": response,
            "sources": sources
        })

        # Single save point — UPDATE existing session or INSERT new one
        st.session_state["active_session_id"] = save_session(
            st.session_state.messages,
            session_id=st.session_state["active_session_id"]
        )

def main():
    st.set_page_config(
        page_title="ResearchMind",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={}
    )

    apply_custom_css()

    with st.sidebar:
        st.title("🧠 ResearchMind")
        page = st.radio("Navigate", ["Chat", "Instructions"], index=0)

    if page == "Instructions":
        instructions_page()
    if page == "Chat":
        chat_page()


if __name__ == "__main__":
    main()
