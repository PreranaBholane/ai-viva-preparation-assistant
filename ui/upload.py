"""
Study Material Management UI.
Handles multi-format document uploads (PDF, PPTX, DOCX, TXT), text extraction,
cleaning, chunking, FAISS vector indexing, and knowledge base clearing.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import config
from utils.file_utils import (
    save_single_uploaded_file,
    save_uploaded_files,
    get_stored_documents,
    clear_uploaded_documents,
    load_sample_material_to_uploads,
    format_file_size,
)
from rag.document_loader import load_multiple_documents
from rag.document_processor import process_documents
from rag.chunking import split_documents, get_chunk_statistics
from rag.faiss_store import faiss_manager


def render_upload_page():
    """
    Render the Study Material Management interface.
    Operates in single-document mode: each upload replaces prior files
    so only the chosen document is processed and indexed.
    """
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">📄 Study Material Management</h1>
            <p class="hero-subtitle">Upload your course notes, slides, question banks, or textbooks to create a local FAISS vector knowledge base.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📤 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose file (PDF, PPTX, DOCX, TXT)",
            type=["pdf", "pptx", "ppt", "docx", "doc", "txt"],
            accept_multiple_files=False,
            help="Upload a single lecture note, presentation, or study document.",
        )

        if uploaded_file is not None:
            last_uploaded = st.session_state.get("staged_filename")
            if last_uploaded != uploaded_file.name:
                # Save single file and clear any previously staged files
                save_single_uploaded_file(uploaded_file, replace_existing=True)
                st.session_state["staged_filename"] = uploaded_file.name
                st.session_state["faiss_ready"] = False
                st.session_state.pop("last_chunk_stats", None)
                st.success(f"✅ Staged **'{uploaded_file.name}'** (previous files cleared). Ready to process!")
                st.rerun()

    with col2:
        st.subheader("⚡ Quick Actions")
        st.markdown(
            "Don't have documents ready? Load sample **Machine Learning Viva Notes** with 1-click."
        )
        if st.button("📥 Load Sample Study Material", use_container_width=True):
            copied = load_sample_material_to_uploads(single_file=True)
            if copied:
                st.session_state["staged_filename"] = copied[0].name
                st.session_state["faiss_ready"] = False
                st.session_state.pop("last_chunk_stats", None)
                st.success(f"Loaded sample file **'{copied[0].name}'**!")
                st.rerun()
            else:
                st.warning("No sample files found in data/sample_materials.")

    st.markdown("---")

    # Display Current Single Staged Document
    st.subheader("📁 Staged Document for Ingestion")
    current_docs = get_stored_documents()

    if not current_docs:
        st.info("No document is currently staged. Upload a file above or click **Load Sample Study Material**.")
    else:
        # Guarantee single document mode
        active_doc = current_docs[0]
        stat = active_doc.stat()
        file_data = [{
            "Filename": active_doc.name,
            "Format": active_doc.suffix.upper().replace(".", ""),
            "Size": format_file_size(stat.st_size),
            "Status": "Ready for Processing",
        }]
        df_files = pd.DataFrame(file_data)
        st.dataframe(df_files, use_container_width=True, hide_index=True)

        col_proc, col_clear = st.columns([1, 1])

        with col_proc:
            if st.button("🚀 Process Document & Build FAISS Index", type="primary", use_container_width=True):
                with st.spinner(f"Processing '{active_doc.name}' through RAG pipeline..."):
                    try:
                        # 1. Load the single active document
                        load_res = load_multiple_documents([active_doc])
                        raw_docs = load_res["documents"]

                        if not raw_docs:
                            st.error(f"No valid text could be extracted from '{active_doc.name}'.")
                            return

                        # 2. Clean and normalize
                        cleaned_docs = process_documents(raw_docs)

                        # 3. Chunk document
                        chunks = split_documents(
                            cleaned_docs,
                            chunk_size=config.CHUNK_SIZE,
                            chunk_overlap=config.CHUNK_OVERLAP,
                        )

                        chunk_stats = get_chunk_statistics(chunks)

                        # 4. Generate embeddings and create local FAISS index
                        active_key = config.get_openai_api_key(
                            st.session_state.get("custom_api_key")
                        )
                        try:
                            faiss_manager.create_index(chunks, api_key=active_key)
                        except Exception as create_err:
                            err_str = str(create_err).lower()
                            if "credit_balance_exhausted" in err_str or "insufficient_quota" in err_str or "429" in err_str:
                                from rag.embeddings import LocalOfflineEmbeddings
                                faiss_manager.create_index(chunks, embeddings=LocalOfflineEmbeddings())
                                st.info("ℹ️ **Notice:** OpenAI account has $0 credit balance. Switched to high-performance local offline embeddings so you can test seamlessly.")
                            else:
                                raise create_err

                        st.session_state["faiss_ready"] = True
                        st.session_state["last_chunk_stats"] = chunk_stats

                        st.balloons()
                        st.success(
                            f"✅ **Successfully indexed {len(chunks)} chunks** from '{active_doc.name}' into FAISS!"
                        )

                        # Display Extraction Breakdown
                        st.markdown("### 📊 Ingestion Breakdown")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Total Chunks", chunk_stats["total_chunks"])
                        m2.metric("Avg Chunk Size", f"{chunk_stats['avg_chunk_size']} chars")
                        doc_disp = active_doc.name if len(active_doc.name) <= 22 else active_doc.name[:19] + "..."
                        m3.metric("Document", doc_disp)
                        m4.metric("Indexed Chunks", faiss_manager.get_index_stats().get("total_vectors", len(chunks)))

                    except Exception as e:
                        st.error(f"❌ Failed to process document: {str(e)}")

        with col_clear:
            if st.button("🗑️ Clear Staged Document & Knowledge Base", use_container_width=True):
                cleared_files = clear_uploaded_documents()
                faiss_manager.clear_index()
                st.session_state["faiss_ready"] = False
                st.session_state.pop("last_chunk_stats", None)
                st.session_state.pop("staged_filename", None)
                st.warning("Cleared staged document and reset the local FAISS index.")
                st.rerun()

    # Knowledge Base Status Section
    st.markdown("---")
    st.subheader("🧠 Local FAISS Knowledge Base Status")
    stats = faiss_manager.get_index_stats()

    kcol1, kcol2, kcol3 = st.columns(3)
    with kcol1:
        if stats["index_exists"]:
            st.markdown("🟢 **Status:** Ready for RAG")
        else:
            st.markdown("🔴 **Status:** No Index Found")
    with kcol2:
        st.markdown(f"📦 **Total Vectors:** `{stats['total_vectors']}`")
    with kcol3:
        st.markdown(f"💾 **Storage Size:** `{stats['faiss_file_size_kb']} KB (FAISS)`")
