"""
Automated Verification Suite for the AI Viva Preparation Assistant.
Tests document loaders, text cleaner, chunker, FAISS manager, retriever,
and Mock Viva evaluation logic.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from rag.document_loader import load_single_document, load_txt
from rag.document_processor import clean_text, process_documents
from rag.chunking import split_documents, get_chunk_statistics
from rag.faiss_store import FAISSStoreManager
from rag.retriever import VivaRetriever
from viva.question_generator import extract_json_from_response
from viva.feedback import VivaFeedbackCompiler
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class DummyDeterministicEmbeddings(Embeddings):
    """
    Deterministic pseudo-embeddings for testing FAISS pipeline offline
    without requiring active OpenAI API credits. Uses simple term frequency vector.
    """

    VOCAB = [
        "random", "forest", "ensemble", "cnn", "convolution", "neural",
        "supervised", "learning", "data", "tree", "bfs", "dfs",
        "overfitting", "pruning", "feature", "classification"
    ]

    def _text_to_vector(self, text: str):
        t_lower = text.lower()
        vec = []
        for word in self.VOCAB:
            count = t_lower.count(word)
            vec.append(float(count * 10.0))
        # Add small base noise
        return [v + 0.001 for v in vec]

    def embed_documents(self, texts):
        return [self._text_to_vector(t) for t in texts]

    def embed_query(self, text):
        return self._text_to_vector(text)


class TestAIVivaPipeline(unittest.TestCase):
    """
    Unit and integration tests for RAG and Viva submodules.
    """

    def setUp(self):
        self.sample_notes_path = config.SAMPLE_DIR / "Machine_Learning_Viva_Notes.txt"
        self.test_store_dir = config.BASE_DIR / "vectorstore_test"
        self.test_store_dir.mkdir(parents=True, exist_ok=True)
        self.dummy_emb = DummyDeterministicEmbeddings()

    def tearDown(self):
        # Cleanup test vectorstore
        import shutil

        if self.test_store_dir.exists():
            shutil.rmtree(self.test_store_dir)

    def test_sample_file_exists(self):
        """Verify sample study notes exist and are readable."""
        self.assertTrue(self.sample_notes_path.exists(), "Sample study notes file must exist.")
        self.assertGreater(self.sample_notes_path.stat().st_size, 500)

    def test_document_loader_txt(self):
        """Test TXT loader and section extraction."""
        docs = load_txt(self.sample_notes_path)
        self.assertGreater(len(docs), 0, "Loader should extract sections from sample notes.")
        first_doc = docs[0]
        self.assertIn("source", first_doc.metadata)
        self.assertEqual(first_doc.metadata["file_type"], "TXT")
        self.assertIn("Supervised Learning", first_doc.page_content)

    def test_document_loaders_pdf_pptx_docx(self):
        """Test PDF, PPTX, and DOCX extractors."""
        from rag.document_loader import load_pdf, load_pptx, load_docx

        pdf_path = config.SAMPLE_DIR / "Machine_Learning_Viva_Notes.pdf"
        docx_path = config.SAMPLE_DIR / "Machine_Learning_Viva_Notes.docx"
        pptx_path = config.SAMPLE_DIR / "Machine_Learning_Viva_Slides.pptx"

        if pdf_path.exists():
            pdf_docs = load_pdf(pdf_path)
            self.assertGreater(len(pdf_docs), 0)
            self.assertEqual(pdf_docs[0].metadata["file_type"], "PDF")

        if docx_path.exists():
            docx_docs = load_docx(docx_path)
            self.assertGreater(len(docx_docs), 0)
            self.assertEqual(docx_docs[0].metadata["file_type"], "DOCX")

        if pptx_path.exists():
            pptx_docs = load_pptx(pptx_path)
            self.assertGreater(len(pptx_docs), 0)
            self.assertEqual(pptx_docs[0].metadata["file_type"], "PPTX")

    def test_text_cleaner(self):
        """Test text normalization and hyphen repair."""
        raw = "This is a machine learn-\ning algo-\nrithm with   extra   spaces\n\n\n\nand lines."
        cleaned = clean_text(raw)
        self.assertNotIn("learn-\ning", cleaned)
        self.assertIn("learning", cleaned)
        self.assertIn("algorithm", cleaned)
        self.assertNotIn("   ", cleaned)

    def test_chunking_and_metadata(self):
        """Test recursive character text chunking."""
        docs = load_txt(self.sample_notes_path)
        cleaned = process_documents(docs)
        chunks = split_documents(cleaned, chunk_size=500, chunk_overlap=100)

        self.assertGreater(len(chunks), len(docs))
        stats = get_chunk_statistics(chunks)
        self.assertGreater(stats["total_chunks"], 0)
        self.assertGreater(stats["avg_chunk_size"], 0)

        for c in chunks:
            self.assertIn("chunk_id", c.metadata)
            self.assertIn("source", c.metadata)

    def test_faiss_lifecycle(self):
        """Test FAISS index creation, persistence, loading, searching, and clearing."""
        docs = [
            Document(
                page_content="Random Forest is an ensemble learning method consisting of decision trees.",
                metadata={"source": "ml_notes.pdf", "page": 1, "topic": "Random Forest"},
            ),
            Document(
                page_content="Convolutional Neural Networks (CNNs) use convolution, pooling, and dense layers for image classification.",
                metadata={"source": "deep_learning.pdf", "page": 4, "topic": "CNN"},
            ),
        ]

        manager = FAISSStoreManager(vectorstore_dir=self.test_store_dir)

        # 1. Create & Save Index
        manager.create_index(docs, embeddings=self.dummy_emb)
        self.assertTrue(manager.is_index_present())

        # 2. Load Index in fresh manager
        fresh_manager = FAISSStoreManager(vectorstore_dir=self.test_store_dir)
        store = fresh_manager.load_index(embeddings=self.dummy_emb)
        self.assertIsNotNone(store)

        # 3. Search Index
        results = fresh_manager.search_index("What is Random Forest?", k=1, embeddings=self.dummy_emb)
        self.assertEqual(len(results), 1)
        doc, score = results[0]
        self.assertIn("Random Forest", doc.page_content)

        # 4. Clear Index
        cleared = fresh_manager.clear_index()
        self.assertTrue(cleared)
        self.assertFalse(fresh_manager.is_index_present())

    def test_json_extractor_helper(self):
        """Test safe JSON parsing from LLM markdown code blocks."""
        raw_markdown_json = """```json
[
  {
    "id": 1,
    "question": "What is Supervised Learning?",
    "difficulty": "Easy",
    "topic": "Foundations",
    "ideal_answer": "Supervised learning uses labeled datasets.",
    "key_points": ["Labeled data", "Mapping inputs to outputs"]
  }
]
```"""
        parsed = extract_json_from_response(raw_markdown_json)
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["question"], "What is Supervised Learning?")

    def test_feedback_compiler_offline(self):
        """Test final mock viva feedback compilation."""
        compiler = VivaFeedbackCompiler()
        evaluations = [
            {
                "question_index": 1,
                "question": "What is Random Forest?",
                "topic": "Ensemble Learning",
                "overall_score": 9.0,
                "correctness": 9.0,
                "relevance": 9.0,
                "completeness": 9.0,
                "clarity": 9.0,
                "feedback": "Excellent answer.",
            },
            {
                "question_index": 2,
                "question": "Explain Decision Trees Pruning",
                "topic": "Decision Trees",
                "overall_score": 6.0,
                "correctness": 6.0,
                "relevance": 6.5,
                "completeness": 5.5,
                "clarity": 6.0,
                "feedback": "Needs more detail on pre-pruning vs post-pruning.",
            },
        ]

        report = compiler.generate_final_report(evaluations, api_key="")
        self.assertEqual(report["total_questions"], 2)
        self.assertEqual(report["overall_score"], 7.5)
        self.assertEqual(report["percentage"], 75.0)
        self.assertIn("Ensemble Learning", report["strong_areas"])
        self.assertIn("Decision Trees", report["needs_improvement"])


if __name__ == "__main__":
    unittest.main()
