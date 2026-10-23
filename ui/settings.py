"""
Settings UI module for managing API keys, model choices, RAG chunking parameters,
and FAISS diagnostics.
"""

import os
import streamlit as st
import config
from rag.faiss_store import faiss_manager
from rag.embeddings import validate_embedding_connection


def render_settings_page():
    """
    Render the Settings and Diagnostics page.
    """
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">⚙️ Application Settings & Diagnostics</h1>
            <p class="hero-subtitle">Configure your OpenAI credentials, fine-tune RAG chunking parameters, and monitor FAISS vector store health.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. API Configuration
    st.subheader("🔑 OpenAI API Credentials")
    env_key = os.getenv("OPENAI_API_KEY", "")
    has_env = bool(env_key and env_key.strip())

    if has_env:
        st.success("✅ `OPENAI_API_KEY` detected from `.env` environment file.")
    else:
        st.warning("⚠️ No `OPENAI_API_KEY` found in `.env`. You can enter it below for this session.")

    user_api_key = st.text_input(
        "OpenAI API Key (Session Override)",
        value=st.session_state.get("custom_api_key", ""),
        type="password",
        placeholder="sk-...",
        help="Entering a key here will override the .env key during this session.",
    )

    if user_api_key:
        st.session_state["custom_api_key"] = user_api_key.strip()

    col_test, _ = st.columns([1, 2])
    with col_test:
        if st.button("🔌 Test API Connection"):
            active_key = config.get_openai_api_key(st.session_state.get("custom_api_key"))
            if not active_key:
                st.error("No API key available to test.")
            else:
                with st.spinner("Testing OpenAI Embeddings connection..."):
                    valid, message = validate_embedding_connection(active_key)
                    if valid:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")

    st.markdown("---")

    # 2. Model Configuration
    st.subheader("🤖 Model & RAG Hyperparameters")
    c1, c2 = st.columns(2)

    with c1:
        model_options = [
            "gpt-5.6-luna",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
        ]
        curr_model = st.session_state.get("custom_model", config.OPENAI_MODEL)
        if curr_model not in model_options:
            model_options.insert(0, curr_model)

        selected_model = st.selectbox(
            "Primary LLM Model",
            options=model_options,
            index=model_options.index(curr_model) if curr_model in model_options else 0,
            help="Configured to GPT-5.6 Luna with fallback options.",
        )
        st.session_state["custom_model"] = selected_model

        emb_options = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
        curr_emb = config.EMBEDDING_MODEL
        selected_emb = st.selectbox(
            "Embedding Model",
            options=emb_options,
            index=emb_options.index(curr_emb) if curr_emb in emb_options else 0,
        )

    with c2:
        top_k_val = st.slider(
            "Top-K Chunks to Retrieve",
            min_value=1,
            max_value=10,
            value=config.TOP_K,
            help="Number of most relevant document chunks sent into the prompt context.",
        )
        config.TOP_K = top_k_val

        chunk_size_val = st.slider(
            "Chunk Size (Characters)",
            min_value=300,
            max_value=2000,
            value=config.CHUNK_SIZE,
            step=100,
        )
        config.CHUNK_SIZE = chunk_size_val

        chunk_overlap_val = st.slider(
            "Chunk Overlap (Characters)",
            min_value=50,
            max_value=500,
            value=config.CHUNK_OVERLAP,
            step=25,
        )
        config.CHUNK_OVERLAP = chunk_overlap_val

    st.markdown("---")

    # 3. FAISS Diagnostics
    st.subheader("📁 FAISS Vector Index Health")
    stats = faiss_manager.get_index_stats()

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("FAISS Index File", "Found" if stats["index_exists"] else "Missing")
    d2.metric("Total Vectors", stats["total_vectors"])
    d3.metric("FAISS Size", f"{stats['faiss_file_size_kb']} KB")
    d4.metric("Pickle Meta Size", f"{stats['pkl_file_size_kb']} KB")

    st.markdown(f"**Index Directory Path:** `{stats['storage_path']}`")
