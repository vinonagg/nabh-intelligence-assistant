import os
import hashlib
import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.ingestion import extract_pdf_pages, chunk_pages
from src.retrieval import HybridRetriever
from src.answering import answer_question
from src.graph_store import Neo4jGraphStore

load_dotenv()
st.set_page_config(
    page_title="NABH Intelligence | Assistant",
    page_icon="✚",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = Path(__file__).resolve().parent
BASE_PDF_PATH = PROJECT_DIR / "knowledge_base" / "NABH_Hospital_Accreditation_Standards.pdf"

def configured_api_key():
    """Prefer Streamlit Cloud Secrets; fall back to local environment/.env."""
    try:
        value = st.secrets.get("OPENAI_API_KEY", "")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "").strip()

st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--ink:#142b3b;--muted:#6d7e8e;--line:#e4ebef;--teal:#087f8c;--deep:#092f40;--paper:#f4f7f8;--white:#fff;--gold:#c69a54}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:var(--paper);color:var(--ink)}
.block-container{max-width:1500px;padding:1.7rem 2.5rem 7rem}
[data-testid="stHeader"]{background:transparent}
[data-testid="stToolbar"]{right:1rem}
#MainMenu,footer{visibility:hidden}
[data-testid="stSidebar"]{background:#0b2636;border-right:1px solid #193c4b}
[data-testid="stSidebar"]>div:first-child{padding:1.6rem 1.15rem}
[data-testid="stSidebar"] *{color:#eaf2f5}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#c2d3db!important}
[data-testid="stSidebar"] hr{border-color:#254453}
[data-testid="stSidebar"] label{color:#d8e6eb!important;font-size:.83rem;font-weight:600}
[data-testid="stSidebar"] input{background:#123544;color:#fff;border:1px solid #315260;border-radius:10px}
[data-testid="stSidebar"] [data-baseweb="select"]>div{background:#123544;border-color:#315260;border-radius:10px}
.brand-lockup{display:flex;align-items:center;gap:12px;padding:.35rem 0 1.15rem;border-bottom:1px solid #244654;margin-bottom:1.4rem}
.brand-mark{display:flex;align-items:center;justify-content:center;width:42px;height:42px;border-radius:13px;background:linear-gradient(145deg,#b6f0e6,#4bb9b0);color:#083242;font-size:1.35rem;font-weight:800;box-shadow:0 7px 18px #061d2a}
.brand-name{font-family:'Manrope',sans-serif;font-size:1.02rem;font-weight:800;letter-spacing:-.035em;color:#fff}
.brand-sub{font-size:.66rem;color:#8fb0bc;letter-spacing:.13em;text-transform:uppercase;margin-top:2px}
.sidebar-label{font-size:.65rem;letter-spacing:.16em;text-transform:uppercase;color:#86d4d2!important;font-weight:800;margin:.5rem 0 .6rem}
.hero{position:relative;overflow:hidden;border-radius:26px;padding:2.6rem 2.8rem 2.25rem;background:linear-gradient(112deg,#082b3d 0%,#0a4657 54%,#087f80 100%);box-shadow:0 22px 60px rgba(9,47,64,.16);color:#fff;margin-bottom:1.25rem;isolation:isolate}
.hero:before{content:"";position:absolute;z-index:-1;right:-100px;top:-270px;width:590px;height:590px;border-radius:50%;border:1px solid rgba(255,255,255,.16);box-shadow:0 0 0 45px rgba(255,255,255,.035),0 0 0 100px rgba(255,255,255,.025),0 0 0 160px rgba(255,255,255,.018)}
.hero:after{content:"✳";position:absolute;right:8%;top:18%;font-size:8rem;line-height:1;color:rgba(255,255,255,.055);font-weight:400}
.hero-eyebrow{display:inline-flex;align-items:center;gap:8px;color:#a8eee4;text-transform:uppercase;letter-spacing:.19em;font-size:.67rem;font-weight:800;margin-bottom:1.05rem}
.hero-dot{width:7px;height:7px;border-radius:50%;background:#8fe5d5;box-shadow:0 0 0 5px rgba(143,229,213,.12)}
.hero h1{font-family:'Manrope',sans-serif;color:#fff!important;-webkit-text-fill-color:#fff!important;font-size:clamp(2.25rem,4vw,3.35rem);line-height:1.06;letter-spacing:-.065em;font-weight:800;margin:0 0 .85rem;max-width:760px;opacity:1!important;text-shadow:0 2px 12px rgba(0,0,0,.16)}
.hero p{color:#e0edf0!important;-webkit-text-fill-color:#e0edf0!important;font-size:1rem;line-height:1.75;max-width:700px;margin:0;opacity:1!important}
.hero-bottom{display:flex;flex-wrap:wrap;gap:9px;margin-top:1.6rem}
.hero-tag{border:1px solid rgba(205,244,239,.25);background:rgba(255,255,255,.075);padding:.48rem .78rem;border-radius:999px;font-size:.72rem;color:#effaf8}
.section-kicker{font-size:.66rem;text-transform:uppercase;letter-spacing:.17em;font-weight:800;color:#617784;margin:.45rem 0 .85rem}
.metric-card{position:relative;overflow:hidden;background:#fff;border:1px solid #e2e9ed;border-radius:18px;padding:1.2rem 1.25rem;min-height:126px;box-shadow:0 5px 18px rgba(17,51,67,.035)}
.metric-card:after{content:"";position:absolute;right:-26px;bottom:-43px;width:100px;height:100px;border-radius:50%;background:#edf8f6}
.metric-label{position:relative;z-index:1;color:#647b88;font-size:.75rem;font-weight:700}
.metric-value{position:relative;z-index:1;color:#122e3e;font-family:'Manrope',sans-serif;font-size:1.85rem;font-weight:800;letter-spacing:-.055em;line-height:1.25;margin-top:.55rem}
.metric-note{position:relative;z-index:1;color:#647b88;font-size:.69rem;margin-top:.32rem}
.panel{background:#fff;border:1px solid #e1e9ed;border-radius:20px;padding:1.35rem 1.4rem;box-shadow:0 6px 24px rgba(16,48,64,.035)}
.panel-heading{display:flex;align-items:flex-start;gap:12px;margin-bottom:1.05rem}
.panel-icon{width:39px;height:39px;flex:0 0 39px;border-radius:12px;background:#e6f5f2;color:#087f80;display:flex;align-items:center;justify-content:center;font-size:1.15rem}
.panel-title{font-family:'Manrope',sans-serif;color:#173344;font-size:1.03rem;font-weight:800;letter-spacing:-.025em;margin:.1rem 0 .25rem}
.panel-subtitle{color:#647b88;font-size:.79rem;line-height:1.55}
.stButton button{border-radius:12px!important;border:1px solid #dce7eb!important;background:#fff!important;color:#234353!important;font-weight:650!important;min-height:2.75rem;transition:all .18s ease}
.stButton button:hover{border-color:#83c9c2!important;background:#f3fbf9!important;color:#087f80!important;transform:translateY(-1px)}
.stButton button[kind="primary"]{background:#087f80!important;color:white!important;border-color:#087f80!important}
[data-testid="stChatMessage"]{background:#fff;border:1px solid #e2eaee;border-radius:18px;padding:1.05rem 1.2rem;margin-bottom:.8rem;box-shadow:0 5px 18px rgba(16,48,64,.03)}
[data-testid="stChatInput"]{border-radius:17px;border:1px solid #d9e5e9;box-shadow:0 8px 30px rgba(16,48,64,.06);background:#fff}
[data-testid="stChatInput"] textarea{color:#183747!important;-webkit-text-fill-color:#183747!important}
[data-testid="stChatInput"] textarea::placeholder{color:#71838e!important;-webkit-text-fill-color:#71838e!important;opacity:1!important}
[data-testid="stExpander"]{border:1px solid #e1e9ed!important;border-radius:13px!important;background:#fbfcfd}
[data-testid="stAlert"]{border-radius:13px}
.stTextInput input,.stSelectbox div[data-baseweb="select"]>div{border-radius:11px}
.footer-note{font-size:.7rem;line-height:1.65;color:#c2d3db;padding:.5rem 0}
.empty-state{background:linear-gradient(180deg,#fff,#f9fbfc);border:1px dashed #cddde2;border-radius:18px;padding:1.7rem;text-align:center}
.empty-state-title{font-family:'Manrope',sans-serif;font-weight:800;font-size:1.05rem;color:#1b3b4b;margin-bottom:.4rem}
.empty-state-copy{font-size:.82rem;color:#647b88;line-height:1.65}
.stApp,.stApp p,.stApp label,.stApp [data-testid="stMarkdownContainer"] p{color:#243e4c}
.stApp .stCaption,.stApp [data-testid="stCaptionContainer"],.stApp small{color:#617784!important}
.stApp h1,.stApp h2,.stApp h3,.stApp h4{color:#153647}
.hero h1,.hero h2,.hero h3{color:#fff!important;-webkit-text-fill-color:#fff!important;opacity:1!important}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,[data-testid="stSidebar"] .stCaption,[data-testid="stSidebar"] [data-testid="stCaptionContainer"],[data-testid="stSidebar"] small{color:#c2d3db!important}
[data-testid="stStatusWidget"] *{color:#294655!important}
.metric-note,.metric-label,.panel-subtitle,.empty-state-copy{color:#647b88!important}
.hero p{color:#e0edf0!important}
@media(max-width:900px){.block-container{padding:1rem 1rem 5rem}.hero{padding:1.65rem}.hero h1{font-size:2.15rem}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-eyebrow"><span class="hero-dot"></span> Clinical quality intelligence platform</div>
  <h1>Make accreditation<br>readiness measurable.</h1>
  <p>An evidence-led workspace for hospital quality teams to interpret NABH standards, prepare audit evidence and strengthen patient-safety systems—with source-linked answers, not guesswork.</p>
  <div class="hero-bottom">
    <span class="hero-tag">✦ Hybrid retrieval</span>
    <span class="hero-tag">↗ Page-level evidence</span>
    <span class="hero-tag">◈ Quality & patient safety</span>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div class="brand-lockup">
      <div class="brand-mark">N</div>
      <div><div class="brand-name">NABH Intelligence</div><div class="brand-sub">Accreditation workspace</div></div>
    </div>
    <div class="sidebar-label">Workspace configuration</div>
    """, unsafe_allow_html=True)
    st.markdown("**Knowledge base file**")
    st.caption("On Streamlit Cloud, provide the licensed PDF in the app repository or use authorized storage. Locally, place it in the project folder.")
    st.markdown("<div style='background:#173b4b;border:1px solid #3b6472;border-radius:10px;padding:10px 12px;color:#edf7fa;font-size:.78rem;line-height:1.5;overflow-wrap:anywhere'>knowledge_base/<br><b>NABH_Hospital_Accreditation_Standards.pdf</b></div>", unsafe_allow_html=True)
    if BASE_PDF_PATH.exists():
        st.success("Base PDF detected")
    else:
        st.warning("No source PDF available in this deployment.")
    if st.button("Reload knowledge base", use_container_width=True):
        for key in ("retriever", "index_key", "source_name", "page_count", "chunk_count"):
            st.session_state.pop(key, None)
        st.rerun()
    key_from_secrets = configured_api_key()
    api_key = st.text_input(
        "OpenAI API key",
        value=key_from_secrets,
        type="password",
        help="Cloud: add OPENAI_API_KEY under Streamlit app Settings → Secrets. Local: use .env.",
    )
    model = st.selectbox("Answer model", ["gpt-4o-mini", "gpt-4.1-mini"], index=0)
    top_k = st.slider("Evidence passages", min_value=3, max_value=12, value=7)
    st.markdown("---")
    st.markdown("<div class='sidebar-label'>Retrieval settings</div>", unsafe_allow_html=True)
    use_graph = st.checkbox("Enable Neo4j graph retrieval", value=False)
    st.caption("Vector + BM25 retrieval works without Neo4j. Graph lookup requires valid credentials in Streamlit Secrets or local .env.")
    st.markdown("---")
    st.markdown("<div class='footer-note'>🔒 <b>Data handling</b><br>PDF extraction and indexing run in the app runtime. Selected passages are sent to your configured OpenAI model when you ask a question. Do not upload identifiable patient information.</div>", unsafe_allow_html=True)
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if not api_key:
    st.info("Add OPENAI_API_KEY in Streamlit app Secrets, or enter it in the sidebar to start.")
    st.stop()

if not BASE_PDF_PATH.exists():
    st.markdown("""
    <div class="empty-state">
      <div style="font-size:2rem;margin-bottom:.6rem;color:#087f80">▤</div>
      <div class="empty-state-title">Knowledge base document not connected</div>
      <div class="empty-state-copy">The app code is deployed, but this cloud instance cannot read files stored only on your Mac. Add the authorized standards PDF to the server's <b>knowledge_base</b> directory, or configure an approved private document source. The PDF is intentionally not bundled with this public repository.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

digest = hashlib.sha256(BASE_PDF_PATH.read_bytes()).hexdigest()
cache_key = f"nabh_base_index_{digest}_{model}"
if st.session_state.get("index_key") != cache_key:
    with st.status("Reading the NABH base PDF and preparing the knowledge index…", expanded=True) as status:
        try:
            pages = extract_pdf_pages(str(BASE_PDF_PATH))
            chunks = chunk_pages(pages)
            st.write(f"Read {len(pages)} pages and prepared {len(chunks)} passages.")
            retriever = HybridRetriever(api_key=api_key, model=model)
            retriever.build(chunks)
            st.session_state["retriever"] = retriever
            st.session_state["index_key"] = cache_key
            st.session_state["source_name"] = BASE_PDF_PATH.name
            st.session_state["page_count"] = len(pages)
            st.session_state["chunk_count"] = len(chunks)
            status.update(label="Base knowledge file indexed", state="complete", expanded=False)
        except Exception as exc:
            st.error(f"Could not index the base PDF: {exc}")
            st.stop()

st.caption(f"Base source: **{BASE_PDF_PATH.name}** · Loaded directly from the project folder")
retriever = st.session_state.get("retriever")
if not retriever:
    st.stop()

st.markdown("<div class='section-kicker'>Workspace overview / Knowledge base status</div>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Source pages</div><div class='metric-value'>{st.session_state.get('page_count', 0):,}</div><div class='metric-note'>Pages extracted from PDF</div></div>", unsafe_allow_html=True)
with m2:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Evidence passages</div><div class='metric-value'>{st.session_state.get('chunk_count', 0):,}</div><div class='metric-note'>Searchable text segments</div></div>", unsafe_allow_html=True)
with m3:
    st.markdown("<div class='metric-card'><div class='metric-label'>Retrieval strategy</div><div class='metric-value' style='font-size:1.15rem'>Hybrid search</div><div class='metric-note'>Semantic + keyword ranking</div></div>", unsafe_allow_html=True)
with m4:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Knowledge graph</div><div class='metric-value' style='font-size:1.15rem'>{'Enabled' if use_graph else 'Optional'}</div><div class='metric-note'>{'Neo4j lookup selected' if use_graph else 'Vector + BM25 available'}</div></div>", unsafe_allow_html=True)

st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)
left, right = st.columns([1.55, 1], gap="large")
with left:
    st.markdown("<div class='panel-heading'><div class='panel-icon'>✦</div><div><div class='panel-title'>Ask the NABH Assistant</div><div class='panel-subtitle'>Explore requirements, audit evidence and readiness practices. Answers are grounded in retrieved passages with page references.</div></div></div>", unsafe_allow_html=True)
with right:
    st.markdown("<div class='panel-heading'><div class='panel-icon'>↗</div><div><div class='panel-title'>Suggested questions</div><div class='panel-subtitle'>Useful starting points for your accreditation workflow.</div></div></div>", unsafe_allow_html=True)
    prompts = [
        "Explain what NABH is in short notes",
        "What evidence should we prepare for an accreditation audit?",
        "How can a hospital identify and close compliance gaps?",
    ]
    for i, prompt in enumerate(prompts):
        if st.button(prompt, key=f"prompt_{i}", use_container_width=True):
            st.session_state.pending_question = prompt

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"View source evidence · {len(message['sources'])} passages"):
                for source in message["sources"]:
                    st.markdown(f"**{source['label']}**")
                    st.caption(f"Retrieval score: {source['score']:.3f}")
                    st.write(source["text"])

question = st.chat_input("Ask about standards, evidence, hospital processes or audit readiness…")
if not question and st.session_state.get("pending_question"):
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    graph_store = None
    if use_graph:
        try:
            graph_store = Neo4jGraphStore.from_env()
        except Exception:
            st.warning("Neo4j is unavailable. Continuing with semantic + keyword retrieval.")
    with st.chat_message("assistant"):
        with st.spinner("Searching standards and preparing a cited answer…"):
            evidence = retriever.search(question, top_k=top_k)
            if graph_store:
                try:
                    graph_evidence = graph_store.search(question, limit=4)
                    evidence = retriever.merge_graph_evidence(evidence, graph_evidence)
                except Exception:
                    st.warning("Graph lookup failed. The answer will use retrieved document evidence only.")
            result = answer_question(question, evidence, api_key=api_key, model=model)
        st.markdown(result["answer"])
        with st.expander(f"View source evidence · {len(result['sources'])} passages", expanded=True):
            for source in result["sources"]:
                st.markdown(f"**{source['label']}**")
                st.caption(f"Retrieval score: {source['score']:.3f}")
                st.write(source["text"])
        st.caption("AI-supported reference only — verify decisions against the current official NABH standards and applicable law.")
    st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
