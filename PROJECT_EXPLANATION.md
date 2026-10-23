# 📚 Comprehensive Project Explanation Guide

> **Project Name:** AI Viva Preparation Assistant  
> **Course / Context:** College Teachers Assessment (TA) Project  
> **Domain:** Artificial Intelligence, Natural Language Processing, Retrieval-Augmented Generation (RAG)

---

## 1. What is the Project?
The **AI Viva Preparation Assistant** is a specialized academic software tool designed to help engineering and computer science students prepare for oral viva examinations. 

Instead of searching through hundreds of pages of lecture slides and textbooks or relying on generic search engines, students upload their syllabus documents. The system processes these materials into a semantic vector index and provides an interactive AI Viva Examiner that answers technical questions, tests the student with realistic viva questions, evaluates oral responses, and pinpoints conceptual weaknesses before the final exam.

---

## 2. Problem Statement
In undergraduate and postgraduate technical courses, viva examinations form a crucial component of grading. However, students face major hurdles:
1. **Lack of Personalized Viva Practice**: Teachers have limited time to conduct one-on-one mock oral exams for every student.
2. **Information Overload**: Notes, slide decks, and textbook chapters are scattered across PDF, PPTX, and DOCX files.
3. **Generic AI Hallucinations**: Standard chatbots (like generic ChatGPT) answer from open web knowledge rather than adhering to the specific course syllabus, textbook definitions, or professor's notes.
4. **Lack of Quantitative Feedback**: Students do not know how to structure a viva answer (defining terms, explaining mechanisms, and providing examples in under 60 seconds).

---

## 3. Project Objectives
1. Build an end-to-end **RAG (Retrieval-Augmented Generation)** pipeline that restricts AI answers strictly to uploaded study materials.
2. Implement **FAISS (Facebook AI Similarity Search)** for fast, local vector search without requiring heavy external database servers (no ChromaDB, SQLite, MySQL, or MongoDB).
3. Support multi-format course materials: **PDF, PPT/PPTX, DOC/DOCX, and TXT**.
4. Develop an interactive **Mock Viva Room** where the AI acts as a viva examiner asking probing follow-up questions.
5. Provide multi-criteria quantitative evaluation: **Correctness, Relevance, Completeness, Clarity, and Overall Score (/10)**.
6. Present a professional, glassmorphic Streamlit web interface with clear source citations (filename, page numbers, exact text chunks).

---

## 4. Proposed Solution
We developed a modular Python application integrating **Streamlit**, **LangChain**, **FAISS**, and **OpenAI GPT-5.6 Luna**:
- **Ingestion**: Documents are extracted, cleaned, and chunked with overlap.
- **Indexing**: OpenAI embeddings are indexed locally in binary FAISS format (`index.faiss` + `index.pkl`).
- **Retrieval**: Questions trigger cosine/L2 nearest-neighbor search to retrieve top relevant chunks.
- **Generation & Defense**: GPT-5.6 Luna synthesizes answers grounded in the retrieved chunks, formatted with viva definitions, explanations, examples, and examiner tips.
- **Evaluation**: The student's responses in Mock Viva are evaluated against the source documents using a structured scoring rubric.

---

## 5. What is RAG (Retrieval-Augmented Generation)?
**RAG** is an AI architecture that combines an information retrieval system with a generative Large Language Model.

Instead of asking an LLM to answer purely from its internal training weights, RAG first searches an external knowledge repository for relevant facts, injects those facts into the prompt as *context*, and instructs the LLM to generate an answer grounded exclusively in that context.

---

## 6. Why RAG?
1. **Zero Hallucination**: Prevents the model from making up definitions or formulas not in the syllabus.
2. **Domain-Specific Grounding**: Adapts instantly to the professor's lecture slides without expensive fine-tuning.
3. **Source Verifiability**: Every answer includes clickable source expanders citing the exact document and page number.
4. **Data Privacy & Cost Efficiency**: Eliminates the need to train or fine-tune models; works on-the-fly with standard LLM APIs.

---

## 7. What is an Embedding?
An **embedding** is a mathematical transformation that converts text (words, sentences, or paragraphs) into a high-dimensional vector of real numbers (e.g., a 1,536-dimensional array).

Embeddings capture semantic meaning rather than exact keyword matches. For example, the vector for *"Random Forest"* is located very close in vector space to *"Ensemble of decision trees"*, allowing the system to find relevant information even if the student uses different phrasing.

---

## 8. What is FAISS?
**FAISS** (Facebook AI Similarity Search) is an open-source C++/Python library developed by Meta AI Research for efficient similarity search and clustering of dense vectors.

It can compare query embeddings against millions of stored document vectors in milliseconds using optimized indexing algorithms like flat L2 distance or Hierarchical Navigable Small World (HNSW) graphs.

---

## 9. Why FAISS Instead of ChromaDB or SQL?
1. **Self-Contained & Lightweight**: FAISS is a pure vector index library, not a heavy client-server database.
2. **Local Persistence**: Saves cleanly to two binary files (`index.faiss` and `index.pkl`) without background database daemons or network socket overhead.
3. **High Performance**: Highly optimized C++ backend with SIMD CPU acceleration (`faiss-cpu`).
4. **Academic Simplicity**: No external database setup or configuration required for examiners running the project.

---

## 10. What is an LLM?
A **Large Language Model (LLM)** is a deep neural network (based on the Transformer architecture) trained on vast text datasets to understand, generate, summarize, and reason over natural language.

---

## 11. Why GPT-5.6 Luna?
1. **High Reasoning & Grounding**: Capable of strictly following academic negative constraints (e.g., *"Do not invent facts not present in context"*).
2. **Structured JSON Output**: Flawlessly outputs parseable JSON schemas for question banks, rubrics, and feedback reports.
3. **Contextual Awareness**: Generates realistic, polite, yet rigorous academic examiner follow-up questions.

---

## 12. Complete Document Processing Flow
```
1. File Upload (PDF, PPTX, DOCX, TXT)
   ↓
2. File Type Detection & Parser Routing
   • PDF -> PyMuPDF (fitz) page-by-page extraction
   • PPTX -> python-pptx slide, shape, and table extraction
   • DOCX -> python-docx paragraph, heading, and table extraction
   • TXT -> Multi-encoding reader (utf-8, latin-1)
   ↓
3. Text Cleaning & Normalization
   • Remove control characters
   • Fix broken hyphenated line-ends
   • Collapse extra whitespace and empty lines
   ↓
4. Semantic Text Chunking
   • RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=200)
   • Metadata enrichment (source filename, page number, topic, chunk_id)
   ↓
5. Vector Embedding
   • OpenAI text-embedding-3-small (1536 dimensions)
   ↓
6. FAISS Local Persistence
   • Inverted/Flat index saved to vectorstore/index.faiss and index.pkl
```

---

## 13. Complete Question-Answering Flow
```
1. Student enters viva question in Streamlit chat.
   ↓
2. Question text is converted to a 1536-dimensional embedding vector.
   ↓
3. FAISS performs similarity search and retrieves Top-K (default K=5) chunks.
   ↓
4. Context assembler formats chunks with source citations and metadata.
   ↓
5. Grounded RAG Prompt is synthesized:
   [System Prompt: Academic Viva Coach] + [Retrieved Context] + [Student Question]
   ↓
6. LLM (GPT-5.6 Luna) generates structured answer (Definition, Explanation, Example, Viva Tips).
   ↓
7. Streamlit UI displays the answer with collapsible "View Retrieved Sources" expanders.
```

---

## 14. Interactive Mock Viva Flow
1. **Setup**: Student chooses Subject, Focus Topic, Difficulty, and Number of Questions (3–10).
2. **Question Generation**: System queries FAISS and creates grounded viva questions.
3. **Oral Defense**: The AI Examiner presents Question #1.
4. **Student Answer**: Student types/dictates their answer.
5. **Instant Evaluation**: RAG evaluator scores the answer across 4 dimensions and provides immediate feedback.
6. **Probing Follow-Up**: AI Examiner asks a targeted follow-up question before moving to the next topic.
7. **Session Scorecard**: Once all questions are complete, the system compiles the final assessment report.

---

## 15. Answer Evaluation Metrics
Student answers are evaluated on a 0.0 to 10.0 scale across 4 core academic criteria:
- **Correctness (/10)**: Technical accuracy, absence of misconceptions, and adherence to scientific definitions.
- **Relevance (/10)**: Directness of the response without irrelevant filler or topic drift.
- **Completeness (/10)**: Coverage of critical components, parameters, or edge cases.
- **Clarity (/10)**: Concise phrasing suitable for a 60-second oral viva answer.
- **Overall Score (/10)**: Weighted holistic evaluation.

---

## 16. Architecture & Modularity
The codebase is strictly modularized for maintainability and grading clarity:
- `app.py`: Streamlit entry point and routing.
- `config.py`: Centralized configuration, environment reading, and session overrides.
- `rag/`: Document loaders, cleaner, chunker, embeddings, FAISS store, retriever, and RAG chain.
- `llm/`: LLM client factory and structured system prompts.
- `viva/`: Question generator, mock viva session controller, answer evaluator, and feedback compiler.
- `ui/`: Modern glassmorphic Streamlit components (Upload, Chatbot, Generator, Mock Viva, Settings).
- `utils/`: File management, validation, and Plotly visualization helpers.

---

## 17. Advantages of this System
1. **Targeted to Specific Syllabus**: Evaluates students on what they were actually taught.
2. **Privacy-First & Local**: Embeddings and documents remain on the local filesystem.
3. **No Database Dependencies**: Runs effortlessly without installing database engines.
4. **Actionable Feedback**: Identifies specific weak topics that require revision before the viva exam.
5. **Aesthetic UI**: Modern, glassmorphic UI with dark mode, interactive charts, and smooth interactions.

---

## 18. Limitations
1. **Text-Centric Parsing**: Complex mathematical equations in non-standard PDF formats or low-resolution scanned image PDFs require OCR.
2. **Oral Speech Interface**: Currently relies on typed input in Streamlit (voice-to-text requires browser microphone integration).
3. **API Dependency**: Requires internet connectivity for OpenAI embedding and LLM calls.

---

## 19. Future Scope
1. **Real-Time Voice Viva**: Integration with OpenAI Whisper (Speech-to-Text) and ElevenLabs (Text-to-Speech) for a completely voice-driven oral exam.
2. **Automated Faculty Dashboard**: Batch evaluation allowing professors to administer online vivas to an entire classroom and export CSV gradebooks.
3. **Diagrammatic Questioning**: Vision-LLM integration allowing the examiner to show a circuit/architecture diagram and ask the student to explain components.
