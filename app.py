import streamlit as st
import requests
import json
import time
import sseclient

API_BASE_URL = "http://localhost:8000"
QUERY_ENDPOINT = f"{API_BASE_URL}/api/v1/chat/query"
STREAM_ENDPOINT = f"{API_BASE_URL}/api/v1/chat/query/stream"

st.set_page_config(
    page_title="NIOS AI Assistant",
    page_icon="📚",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Sidebar: Auth Mode & System Status
# ---------------------------------------------------------------------------
st.sidebar.title("🔐 Access Control & Auth")
auth_mode = st.sidebar.radio(
    "Select Mode:",
    ["Public Visitor (Phase 1)", "Authenticated Student (Phase 2)"]
)

auth_headers = {}
if auth_mode == "Authenticated Student (Phase 2)":
    enrollment = st.sidebar.text_input("Student Enrollment No:", value="270041123456")
    auth_headers = {"Authorization": f"Bearer dev_student_{enrollment}"}
    st.sidebar.success(f"👤 Authenticated: `{enrollment}`")
    st.sidebar.caption("✅ Modules 1–5 Active: Admission, Exams, Results, Fees, Grievances")
else:
    st.sidebar.info("🌐 Browsing as Public Guest (General Circulars & FAQs)")

st.sidebar.divider()
st.sidebar.title("⚙️ Telemetry & Options")
stream_enabled = st.sidebar.checkbox("Enable Real-Time SSE Streaming", value=True)
top_k = st.sidebar.slider("Retrieval Depth (Top-k)", min_value=3, max_value=20, value=5)

try:
    health_resp = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=2)
    if health_resp.status_code == 200:
        st.sidebar.success("🟢 API Gateway: Healthy")
    else:
        st.sidebar.warning("🟡 API Gateway: Degraded")
except Exception:
    st.sidebar.error("🔴 API Gateway: Offline")

# ---------------------------------------------------------------------------
# Header & Phase 1 Guided Category Pills
# ---------------------------------------------------------------------------
st.title("📚 NIOS Conversational AI Assistant")
st.caption("Official grounded support for NIOS circulars, examinations, admission, results, and fee queries.")

try:
    cat_resp = requests.get(f"{API_BASE_URL}/api/v1/faq/categories?lang=en", timeout=2)
    if cat_resp.status_code == 200:
        categories = cat_resp.json().get("data", [])
        st.write("##### 📌 Browse Core Categories (Phase 1):")
        cols = st.columns(len(categories))
        for idx, cat in enumerate(categories):
            with cols[idx]:
                if st.button(f"{cat['name']}", key=f"cat_{cat['id']}"):
                    st.session_state.selected_category = cat['id']
                    st.session_state.selected_category_name = cat['name']
except Exception:
    pass

if "selected_category" in st.session_state:
    st.info(f"Viewing topics under: **{st.session_state.selected_category_name}**")
    try:
        top_resp = requests.get(
            f"{API_BASE_URL}/api/v1/faq/topics?category_id={st.session_state.selected_category}&lang=en",
            timeout=2
        )
        if top_resp.status_code == 200:
            topics = top_resp.json().get("topics", [])
            tcols = st.columns(min(len(topics), 4) if topics else 1)
            for tidx, top in enumerate(topics):
                with tcols[tidx % len(tcols)]:
                    if st.button(f"📖 {top['title']}", key=f"top_{top['id']}"):
                        st.session_state.faq_query = top['title']
    except Exception:
        pass

st.divider()

# ---------------------------------------------------------------------------
# Chat Session History
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "meta" in msg and msg["meta"]:
            st.caption(msg["meta"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📄 Grounded Official Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s.get('document')}** (Section: `{s.get('category')}`) | RRF Score: `{s.get('rrf_score')}`")

# ---------------------------------------------------------------------------
# Query Input & Execution
# ---------------------------------------------------------------------------
input_prompt = st.chat_input("Ask a question about admission, fee payment, hall ticket, or exam dates...")

# Check if triggered via topic click
if "faq_query" in st.session_state and st.session_state.faq_query:
    input_prompt = st.session_state.faq_query
    del st.session_state.faq_query

if input_prompt:
    st.session_state.messages.append({"role": "user", "content": input_prompt})
    with st.chat_message("user"):
        st.markdown(input_prompt)

    with st.chat_message("assistant"):
        t0 = time.time()
        headers = {"Content-Type": "application/json"}
        headers.update(auth_headers)

        if stream_enabled:
            # Server-Sent Events (SSE) Streaming
            placeholder = st.empty()
            full_answer = ""
            sources = []
            cache_hit = False
            retrieval_ms = 0.0

            try:
                resp = requests.post(
                    STREAM_ENDPOINT,
                    json={"query": input_prompt, "top_k": top_k},
                    headers=headers,
                    stream=True,
                    timeout=60
                )
                if resp.status_code == 200:
                    client = sseclient.SSEClient(resp)
                    for event in client.events():
                        if not event.data:
                            continue
                        event_data = json.loads(event.data)
                        evt_type = event_data.get("type")

                        if evt_type == "meta":
                            cache_hit = event_data.get("cache_hit", False)
                            retrieval_ms = event_data.get("retrieval_latency_ms", 0.0)
                            sources = event_data.get("sources", [])
                        elif evt_type == "token":
                            full_answer += event_data.get("delta", "")
                            placeholder.markdown(full_answer + "▌")
                        elif evt_type == "done":
                            placeholder.markdown(full_answer)
                            break

                    total_ms = (time.time() - t0) * 1000
                    badge = "⚡ **Redis Tier-4 Cache Hit**" if cache_hit else "🐢 **Direct Pipeline Execution**"
                    meta_str = f"{badge} | Retrieval: `{retrieval_ms:.2f} ms` | Round-Trip: `{total_ms:.2f} ms`"
                    st.caption(meta_str)

                    if sources:
                        with st.expander("📄 Grounded Official Sources"):
                            for s in sources:
                                st.markdown(f"- **{s.get('document')}** (Section: `{s.get('category')}`) | RRF Score: `{s.get('rrf_score')}`")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_answer,
                        "meta": meta_str,
                        "sources": sources
                    })
                else:
                    st.error(f"Error ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.error(f"SSE Streaming error: {str(e)}")

        else:
            # Synchronous Fallback
            with st.spinner("Processing query..."):
                try:
                    resp = requests.post(
                        QUERY_ENDPOINT,
                        json={"query": input_prompt, "top_k": top_k},
                        headers=headers,
                        timeout=60
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data.get("answer", "")
                        cache_hit = data.get("cache_hit", False)
                        retrieval_ms = data.get("retrieval_latency_ms", 0.0)
                        sources = data.get("sources", [])
                        total_ms = (time.time() - t0) * 1000

                        badge = "⚡ **Redis Tier-4 Cache Hit**" if cache_hit else "🐢 **Direct Pipeline Execution**"
                        meta_str = f"{badge} | Retrieval: `{retrieval_ms:.2f} ms` | Round-Trip: `{total_ms:.2f} ms`"

                        st.markdown(answer)
                        st.caption(meta_str)

                        if sources:
                            with st.expander("📄 Grounded Official Sources"):
                                for s in sources:
                                    st.markdown(f"- **{s.get('document')}** (Section: `{s.get('category')}`) | RRF Score: `{s.get('rrf_score')}`")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "meta": meta_str,
                            "sources": sources
                        })
                    else:
                        st.error(f"Error ({resp.status_code}): {resp.text}")
                except Exception as e:
                    st.error(f"API Error: {str(e)}")