# 🎓 AI Viva Preparation Assistant

An intelligent, RAG-powered (Retrieval-Augmented Generation) oral examination preparation platform for college students and teachers. Built using **Python 3.11+**, **Streamlit**, **LangChain**, **FAISS**, and **OpenAI GPT-5.6 Luna**.

---

## 📌 1. Project Objective

The primary objective of the **AI Viva Preparation Assistant** is to help college students prepare effectively for technical oral viva examinations using their own course study material. 

Students and educators can upload syllabus notes, slide decks, textbooks, or question banks in **PDF, PPTX, DOCX, or TXT** format. The system builds a local, privacy-first **FAISS vector knowledge base** and grounds an AI examiner and chatbot in the retrieved content to eliminate hallucinations and provide high-quality viva coaching.

---

## 🌟 2. Key Features

1. **Multi-Format Document Ingestion**:
   - Native parsing of PDF (`PyMuPDF`), PPT/PPTX (`python-pptx`), DOC/DOCX (`python-docx`), and TXT files.
   - Text cleaning, normalization, metadata tagging (source, page/slide, section, topic), and configurable semantic chunking.
2. **Local FAISS Vector Similarity Search**:
   - Zero traditional database dependency (no ChromaDB, SQLite, MySQL, or MongoDB).
   - Fast, persistent vector indexing stored locally in `vectorstore/index.faiss` and `vectorstore/index.pkl`.
3. **Grounded AI Viva Chatbot**:
   - Strictly answers viva questions using retrieved study material chunks.
   - Outputs definitions, concise explanations, real-world examples, and examiner tips.
   - Expandable source citations displaying file name, page/slide number, and exact text snippet.
4. **Targeted Viva Question Generator**:
   - Generates question banks categorized by Subject, Focus Topic, and Difficulty (`Easy`, `Medium`, `Hard`, `Mixed`).
   - Includes model viva answers and key evaluation points.
   - One-click export to Markdown.
5. **Interactive Mock Viva Room**:
   - Simulates a live oral viva examination with turn-by-turn examiner questions.
   - Dynamic probing follow-up questions generated from student responses and study notes.
6. **Multi-Criteria Answer Evaluation**:
   - Scores answers across **Correctness (/10)**, **Relevance (/10)**, **Completeness (/10)**, and **Clarity (/10)**.
   - Produces immediate strengths, missing points, and ideal model answers.
7. **Final Mock Viva Scorecard & Analytics**:
   - Overall Score (/10), Percentage (%), and Viva Readiness Verdict.
   - Interactive Plotly Radar Competency Chart and Question Distribution Bar Chart.
   - Clear breakdown of Strong Areas vs. Needs Improvement topics with actionable preparation advice.
8. **Settings & Diagnostics**:
   - Runtime OpenAI API Key management, model selection, chunking parameter tuning, and FAISS vector index health inspection.

---

## 🛠️ 3. Technology Stack

| Layer | Technology | Description |
|---|---|---|
| **Language** | Python 3.11+ | Core programming language |
| **Frontend UI** | Streamlit | Reactive, glassmorphic dark-mode web application |
| **RAG Framework** | LangChain | Text splitters, document loaders, and RAG chains |
| **Vector Search** | FAISS (`faiss-cpu`) | Local dense vector similarity search and indexing |
| **LLM** | OpenAI API (`GPT-5.6 Luna`) | High-reasoning language model for viva Q&A & evaluation |
| **Embeddings** | OpenAI `text-embedding-3-small` | 1536-dimensional dense text representations |
| **Document Parsers** | PyMuPDF, python-pptx, python-docx | High-fidelity text extraction across formats |
| **Data & Viz** | Pandas, Plotly | Performance analytics and radar visualizations |

---

## 🏛️ 4. System & RAG Architecture

```
                                  [ STUDENT / USER ]
                                          │
                                          ▼
                             [ STREAMLIT WEB INTERFACE ]
                                    │           │
              (Study Materials Upload)         (Viva Question / Answer)
                                    │           │
                                    ▼           ▼
                      [ TEXT EXTRACTION PIPELINE ]     [ QUESTION EMBEDDING ]
                       • PyMuPDF (PDF)                 (OpenAI text-embedding-3-small)
                       • python-pptx (PPTX)                     │
                       • python-docx (DOCX)                     ▼
                       • Python File I/O (TXT)         [ FAISS SIMILARITY SEARCH ]
                                    │                  (Cosine / L2 Vector Distance)
                                    ▼                           │
                        [ TEXT CHUNKING & CLEANING ]            ▼
                        (RecursiveCharacterSplitter)   [ TOP-K RELEVANT CHUNKS ]
                                    │                           │
                                    ▼                           ▼
                        [ EMBEDDING GENERATION ] ───► [ GROUNDED RAG PROMPT ]
                                    │                  (System + Context + Question)
                                    ▼                           │
                           [ LOCAL FAISS INDEX ]                ▼
                         (index.faiss + index.pkl)     [ OPENAI GPT-5.6 LUNA ]
                                                                │
                                                                ▼
                                                       [ GROUNDED VIVA ANSWER ]
                                                      + [ SOURCE CITATIONS ]
```

---

## ⚙️ 5. How RAG Works in this Project

1. **Document Ingestion**:
   When documents are uploaded, the parser extracts raw text along with metadata (filename, page/slide number).
2. **Text Cleaning & Splitting**:
   Text is normalized and split using `RecursiveCharacterTextSplitter` with `chunk_size = 1000` characters and `chunk_overlap = 200` characters to preserve contextual boundaries.
3. **Vector Embedding**:
   Each chunk is transformed into a dense floating-point vector using OpenAI's embedding model.
4. **FAISS Indexing**:
   Vectors are indexed in a local FAISS index (`vectorstore/index.faiss`) with metadata stored in `vectorstore/index.pkl`.
5. **Semantic Retrieval**:
   When the student asks a question, the question is converted into an embedding and FAISS performs a fast nearest-neighbor search to retrieve the top $K$ most relevant chunks.
6. **Grounded Synthesis**:
   Retrieved chunks are passed as explicit context into a system prompt that directs **GPT-5.6 Luna** to answer strictly from the provided text, citing sources and avoiding hallucinations.

---

## 💡 6. Why FAISS and GPT-5.6 Luna?

- **Why FAISS?**
  - Facebook AI Similarity Search (FAISS) is an ultra-fast, lightweight library for vector indexing that runs completely in-memory and saves directly to disk.
  - Avoids heavyweight external database servers (like ChromaDB, PostgreSQL, MySQL, or MongoDB), making the application self-contained, portable, and ideal for academic assessment.
- **Why GPT-5.6 Luna?**
  - Provides advanced reasoning and factual adherence essential for academic viva coaching.
  - Ensures accurate multi-criteria rubric evaluation (scoring correctness, relevance, completeness, clarity) and structured JSON outputs.

---

## 🚀 7. Installation & Setup Guide

### Prerequisites
- Python 3.11 or higher
- An active OpenAI API Key

### Step 1: Clone or Extract the Project
```bash
cd "AI-Viva-Preparation-Assistant"
```

### Step 2: Set Up a Virtual Environment
**On Windows (PowerShell / Command Prompt):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a `.env` file in the root directory by copying `.env.example`:
```bash
copy .env.example .env
```
Open `.env` and set your OpenAI API key:
```env
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-5.6-luna
FALLBACK_OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
TOP_K=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```
*(Note: You can also enter or update your API key directly inside the application under the **⚙ Settings** tab).*

---

## 🖥️ 8. Running the Application

Launch the Streamlit web application:
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📖 9. User Guide & Workflow

### 1. Upload Study Material
- Navigate to **📄 Upload Material**.
- Upload your PDF, PPTX, DOCX, or TXT notes, or click **"📥 Load Sample Study Material"** to immediately test with pre-packaged Machine Learning notes.
- Click **"🚀 Process Documents & Build FAISS Index"**.
- View chunk statistics and indexing confirmation.

### 2. AI Viva Chat
- Navigate to **💬 AI Viva Chat**.
- Ask questions like: *"What is Random Forest?"*, *"Explain CNN architecture"*, *"What is the difference between BFS and DFS?"*.
- Click on **"📚 View Retrieved Sources"** under each answer to inspect the exact pages and text snippets retrieved.

### 3. Generate Viva Questions
- Navigate to **🎯 Generate Questions**.
- Enter your Subject, Focus Topic, and choose a Difficulty (`Easy`, `Medium`, `Hard`, `Mixed`).
- Download the generated question bank as Markdown or click **"🎤 Practice These in Mock Viva Room"**.

### 4. Interactive Mock Viva
- Navigate to **🎤 Mock Viva**.
- The AI Examiner presents viva questions one by one.
- Type your oral response and click **"📤 Submit Answer to Examiner"**.
- Receive immediate feedback, score breakdown, and probing follow-up questions.
- View the comprehensive final scorecard with Plotly radar charts, strong/weak areas, and examiner revision tips.

---

## 🧪 10. Automated Testing

Run the automated RAG pipeline verification suite:
```bash
python tests/test_pipeline.py
```

---

## 🔮 11. Future Scope

1. **Speech-to-Text & Text-to-Speech Integration**: Real-time voice interaction using Whisper API and ElevenLabs for realistic oral viva simulations.
2. **Diagram & Equation Parsing**: Multi-modal vision processing for mathematical formulas and architectural circuit diagrams.
3. **Teacher Grading Dashboard**: Analytics export enabling professors to evaluate viva transcripts across student batches.
4. **Adaptive Difficulty Engine**: Dynamically adjusting examiner question difficulty based on student response quality.
