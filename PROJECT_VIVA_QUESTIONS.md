# 🎓 30+ Examiner Viva Questions & Answers

> **Project:** AI Viva Preparation Assistant  
> **Purpose:** Use this document to prepare for your project viva examination. These are the most common questions college professors and external examiners will ask about your project.

---

### Q1. What is RAG (Retrieval-Augmented Generation)?
**Answer:**  
RAG is an AI technique that combines an Information Retrieval system with a generative Large Language Model. Instead of relying solely on the LLM's pre-trained knowledge, RAG retrieves relevant text chunks from an external knowledge base (our uploaded study materials) and passes them as context into the prompt so the LLM generates accurate, grounded answers.

---

### Q2. Why did you use RAG in this project instead of just a normal chatbot like ChatGPT?
**Answer:**  
A normal chatbot answers from general internet training data and often hallucinates or gives definitions different from the course textbook. RAG ensures the AI answers strictly from the student's uploaded syllabus notes, cites exact page numbers, and admits when information is missing rather than inventing facts.

---

### Q3. What is FAISS and who developed it?
**Answer:**  
FAISS (Facebook AI Similarity Search) is an open-source C++ library with Python bindings developed by Meta AI Research. It is designed for fast, memory-efficient similarity search and clustering of dense vector embeddings.

---

### Q4. Why did you choose FAISS instead of ChromaDB, SQLite, or MongoDB?
**Answer:**  
1. **Lightweight & Self-Contained:** FAISS is a dedicated vector indexing library with no background database servers or network ports.
2. **Local File Persistence:** It saves directly into two local files (`index.faiss` and `index.pkl`).
3. **Speed & Efficiency:** FAISS is written in highly optimized C++ with CPU vectorization, making it significantly faster and simpler to deploy than traditional databases.

---

### Q5. What are embeddings and what dimension do they have?
**Answer:**  
An embedding is a numerical representation of text in a high-dimensional vector space. Semantically similar texts are placed close to each other. In our project, we use OpenAI's `text-embedding-3-small`, which produces 1,536-dimensional dense vectors.

---

### Q6. What is vector similarity and how is it calculated in FAISS?
**Answer:**  
Vector similarity measures how close two embeddings are in vector space. In FAISS, it is typically computed using Euclidean (L2) distance or Cosine Similarity (Dot Product of normalized vectors). A smaller L2 distance indicates higher conceptual similarity between the student's question and the document chunk.

---

### Q7. What is text chunking and why is it necessary in RAG?
**Answer:**  
Chunking is the process of breaking large documents into smaller, manageable text segments (e.g., 1,000 characters). It is necessary because:
1. LLMs have context window limits.
2. Smaller chunks improve retrieval precision by fetching only the exact paragraphs relevant to the question rather than entire chapters.

---

### Q8. Why do we use chunk overlap (e.g., 200 characters)?
**Answer:**  
Chunk overlap ensures that important sentences or concepts are not split arbitrarily at chunk boundaries. It preserves context across consecutive chunks so no semantic meaning is lost.

---

### Q9. What text splitter did you use and why?
**Answer:**  
We used LangChain's `RecursiveCharacterTextSplitter`. It splits hierarchically on double newlines (`\n\n`), single newlines (`\n`), periods (`.`), and spaces. This keeps paragraphs and complete sentences intact before falling back to word splits.

---

### Q10. What is LangChain's role in this project?
**Answer:**  
LangChain acts as the orchestration framework. It coordinates the document loaders, text chunkers, embedding functions, FAISS vector store interfaces, prompt templates, and LLM invocation pipelines.

---

### Q11. What is an LLM (Large Language Model)?
**Answer:**  
An LLM is a deep learning model based on the Transformer architecture trained on massive textual corpora. It uses self-attention mechanisms to understand syntax, semantics, and context, allowing it to generate coherent human-like text.

---

### Q12. Why did you use GPT-5.6 Luna?
**Answer:**  
GPT-5.6 Luna provides state-of-the-art reasoning, strict adherence to system prompts (reducing hallucination), and reliable generation of structured JSON for rubric evaluations and viva question banks.

---

### Q13. What is AI hallucination and how does your project prevent it?
**Answer:**  
Hallucination occurs when an LLM generates plausible-sounding but factually incorrect information. We prevent it by:
1. Supplying retrieved study notes as ground truth.
2. Using strict system prompts directing the model to state *"Answer not available in the provided study material"* if context is insufficient.
3. Setting temperature to low values (0.2) for factual determinism.

---

### Q14. What happens if a student asks a question not in the uploaded documents?
**Answer:**  
The retriever fetches the closest chunks, but the system prompt instructs the LLM: *"If the retrieved context does not contain enough information, clearly state that the answer is not available in the provided study material."*

---

### Q15. How does the Mock Viva feature work?
**Answer:**  
1. The AI generates topic-specific viva questions from the FAISS knowledge base.
2. The AI acts as an examiner, asking questions one by one.
3. When the student submits an answer, the evaluator assesses it in real time against the study material.
4. The AI asks probing follow-up questions to test deeper understanding.
5. At the end of the session, a full scorecard with Plotly charts is generated.

---

### Q16. What 4 criteria are used to evaluate the student's viva answers?
**Answer:**  
1. **Correctness (/10):** Technical accuracy of definitions and formulas.
2. **Relevance (/10):** Directness of the answer without extraneous drift.
3. **Completeness (/10):** Inclusion of essential components and examples.
4. **Clarity (/10):** Structured, concise delivery suitable for viva.

---

### Q17. How is the final Mock Viva scorecard computed?
**Answer:**  
The system averages scores across all attempted questions, computes percentage, categorizes topics into Strong Areas (≥7.5/10) and Needs Improvement (<7.5/10), and compiles actionable examiner tips for final revision.

---

### Q18. How do you handle PDF files in the ingestion pipeline?
**Answer:**  
We use `PyMuPDF` (`fitz`), which extracts text page by page. It records the page number in the metadata dictionary (`{"source": filename, "page": page_num}`), enabling exact source attribution in the UI.

---

### Q19. How do you extract text from PowerPoint (PPTX) presentations?
**Answer:**  
We use `python-pptx` to iterate over slides, extract text from shapes, titles, and slide tables, and tag each chunk with the corresponding slide number and topic.

---

### Q20. How do you extract text from Microsoft Word (DOCX) files?
**Answer:**  
We use `python-docx` to extract paragraphs and tables, detecting headings as section topic markers and grouping content logically.

---

### Q21. Why did you choose Streamlit for the frontend?
**Answer:**  
Streamlit is a Python-native web framework that allows rapid development of reactive, interactive data science applications with session state management, file uploaders, and chart integration without complex JavaScript frontend overhead.

---

### Q22. Where is the session data stored?
**Answer:**  
All session information (chat messages, active mock viva questions, scores, and evaluations) is stored in Streamlit's in-memory `st.session_state`. No database is needed.

---

### Q23. What files are created when FAISS saves the vector store?
**Answer:**  
1. `index.faiss`: The binary vector index containing the dense numerical embeddings.
2. `index.pkl`: A Python pickle file containing the LangChain `Docstore` and mapping from vector IDs to original text chunks and metadata.

---

### Q24. How does the system allow the user to clear the knowledge base?
**Answer:**  
The `Clear Knowledge Base` button calls `faiss_manager.clear_index()`, which resets the in-memory vector store object and unlinks `index.faiss` and `index.pkl` from disk, along with clearing staged upload files.

---

### Q25. What is the role of Temperature in the LLM configuration?
**Answer:**  
Temperature controls the randomness of token generation:
- Low temperature (0.2): Highly focused, deterministic, and factual (used for Viva QA and answer evaluation).
- Moderate temperature (0.4–0.5): Slightly more creative (used for generating diverse viva questions and follow-up prompts).

---

### Q26. Can you explain the difference between Bagging and Boosting in simple viva terms?
**Answer:**  
- **Bagging (e.g., Random Forest):** Trains multiple weak learners independently in parallel on bootstrap samples and averages their predictions to reduce variance (overfitting).
- **Boosting (e.g., AdaBoost, XGBoost):** Trains learners sequentially where each model corrects the errors of its predecessor to reduce bias.

---

### Q27. What is the difference between BFS and DFS in graph traversal?
**Answer:**  
- **BFS (Breadth-First Search):** Traverses level by level using a FIFO Queue; finds the shortest path in unweighted graphs; space complexity $O(V)$.
- **DFS (Depth-First Search):** Traverses deeply along branches using a LIFO Stack or recursion; used for cycle detection and topological sorting; space complexity $O(H)$.

---

### Q28. What are the key layers in a Convolutional Neural Network (CNN)?
**Answer:**  
1. **Convolutional Layer:** Applies spatial filters to extract local features.
2. **Activation Layer (ReLU):** Introduces non-linearity ($f(x) = \max(0, x)$).
3. **Pooling Layer (Max/Average):** Downsamples feature maps to reduce dimensions and provide translation invariance.
4. **Fully Connected Layer:** Flattens features for final classification.
5. **Softmax Layer:** Converts logits into class probabilities.

---

### Q29. What are the primary limitations of your project?
**Answer:**  
1. Image-only/scanned PDFs without OCR cannot be read directly.
2. Oral viva responses are currently typed or dictated through OS speech input rather than real-time audio streaming.
3. Requires an active OpenAI API key and internet access.

---

### Q30. What would you build next in Future Scope?
**Answer:**  
1. Direct speech-to-speech interaction using OpenAI Whisper and ElevenLabs.
2. Multi-modal vision processing for engineering circuit diagrams and flowcharts.
3. Teacher batch evaluation dashboard with automated class-wide viva grading exports.
