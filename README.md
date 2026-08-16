# RAG PDF Question Answering System

A simple and practical **Retrieval-Augmented Generation (RAG)** application that lets you upload PDF documents and ask questions about their content.

Instead of sending the entire document to an LLM, the application extracts the PDF text, breaks it into meaningful chunks, converts those chunks into embeddings, stores them in a FAISS vector index, and retrieves only the most relevant information when a question is asked.

The retrieved context is then passed to Gemini to generate a grounded answer.

---

## Demo

<p align="center">
  <img src="screenshots/demo.png" alt="RAG PDF Question Answering System" width="900"/>
</p>

> Upload a PDF → Ask a question → Get an answer with the relevant source pages.

---

## What it does

The application supports:

* Uploading one or multiple PDF documents
* Extracting text from PDF files
* Configurable chunk size and overlap
* Semantic search using embeddings
* FAISS-based vector storage
* Top-K relevant chunk retrieval
* Gemini-powered question answering
* Source document and page references
* Chat history
* Document-level information
* Configurable RAG parameters
* Answers grounded in the uploaded documents

If the requested information cannot be found in the uploaded documents, the application is instructed not to invent an answer.

---

## How the RAG pipeline works

```text
                    PDF Upload
                        │
                        ▼
                ┌────────────────┐
                │  PDF Extraction│
                └───────┬────────┘
                        │
                        ▼
                ┌────────────────┐
                │ Text Chunking  │
                └───────┬────────┘
                        │
                        ▼
                ┌────────────────┐
                │   Embeddings   │
                └───────┬────────┘
                        │
                        ▼
                ┌────────────────┐
                │  FAISS Index   │
                └───────┬────────┘
                        │
                  User Question
                        │
                        ▼
                ┌────────────────┐
                │ Semantic Search│
                └───────┬────────┘
                        │
                  Top-K Chunks
                        │
                        ▼
                ┌────────────────┐
                │  Gemini LLM    │
                └───────┬────────┘
                        │
                        ▼
                ┌────────────────┐
                │ Answer + Source│
                └────────────────┘
```

---

## Tech Stack

| Component             | Technology            |
| --------------------- | --------------------- |
| Language              | Python                |
| Frontend              | Streamlit             |
| PDF Processing        | PyMuPDF               |
| Text Splitting        | LangChain             |
| Embeddings            | Sentence Transformers |
| Vector Database       | FAISS                 |
| LLM                   | Google Gemini         |
| Environment Variables | python-dotenv         |

---

## Why RAG?

A normal LLM-based application can answer questions from its training knowledge, but that does not mean it knows the contents of a private PDF uploaded by a user.

RAG solves this by adding a retrieval step before generation.

For every question, the application:

1. Converts the question into an embedding.
2. Searches the FAISS vector index.
3. Finds the most semantically relevant document chunks.
4. Sends those chunks to the LLM as context.
5. Generates an answer based on the retrieved information.

This makes the responses more relevant to the uploaded documents and reduces unnecessary hallucination.

---

## Features

### Multiple PDF Support

Upload multiple documents and search across their contents from the same interface.

### Configurable Chunking

Chunk size and overlap can be adjusted depending on the type of document.

For example:

```text
Chunk Size    → 1000
Chunk Overlap → 200
```

This allows experimentation with retrieval quality.

### Semantic Retrieval

The application does not rely only on exact keyword matching.

Questions and document chunks are converted into vector representations, allowing semantically related content to be retrieved even when the wording is different.

### Source References

Retrieved information keeps track of the original document and page number.

This makes it easier to verify where an answer came from.

### Context-Grounded Answers

The LLM receives the retrieved document context along with the question.

If the required information is not available, the application can respond:

> I couldn't find this information in the uploaded documents.

---

## Project Structure

```text
RAG-PDF-QA/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
│
├── src/
│   ├── pdf_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── llm.py
│   └── rag_pipeline.py
│
├── data/
│   └── sample_pdfs/
│
├── vector_store/
│
└── screenshots/
    ├── home.png
    ├── upload.png
    └── qa.png
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

Do not commit your `.env` file to GitHub.

Make sure `.gitignore` contains:

```text
.env
venv/
__pycache__/
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## Example Questions

After uploading a document, you can ask questions such as:

```text
What is the main objective of this document?

What methodology was used?

What are the key findings?

What limitations were mentioned?

What conclusions were drawn?

What percentage was reported for the proposed approach?

How does the proposed method differ from the baseline?
```

You can also test the system with questions whose answers are **not present in the document**. This is useful for checking whether the application avoids making up information.

---

## RAG Configuration

The application exposes several parameters that affect retrieval performance.

### Chunk Size

Controls how much text is contained in each document chunk.

Smaller chunks can provide more precise retrieval, while larger chunks preserve more surrounding context.

### Chunk Overlap

Maintains some shared text between consecutive chunks.

This helps prevent important information from being lost at chunk boundaries.

### Top-K

Controls the number of chunks retrieved for each question.

For example:

```text
Top-K = 5
```

means the five most relevant chunks are passed to the generation stage.

---

## Example

### Question

```text
What are the main limitations mentioned in the research paper?
```

### Retrieval

```text
research_paper.pdf
Page 8
Page 11
Page 13
```

### Generated Answer

```text
The paper identifies three main limitations:
...

Sources:
research_paper.pdf — Pages 8, 11, 13
```

The source information allows the user to go back to the original document and verify the answer.

---

## Handling Unknown Questions

One of the important parts of this project is testing what happens when the answer does not exist in the uploaded documents.

For example, if the PDF is about machine learning and the user asks:

```text
Who was the first person to walk on Mars?
```

the system should not try to answer using the LLM's general knowledge.

Instead:

```text
I couldn't find this information in the uploaded documents.
```

This is an important part of keeping a RAG application grounded in its retrieved context.

---

## Screenshots

### Home

<p align="center">
  <img src="screenshots/home.png" width="850"/>
</p>

### PDF Upload

<p align="center">
  <img src="screenshots/upload.png" width="850"/>
</p>

### Question Answering

<p align="center">
  <img src="screenshots/qa.png" width="850"/>
</p>

---

## Challenges I Worked Through

Building this project involved more than simply connecting an LLM to a PDF.

Some of the main challenges were:

* Extracting usable text from different PDF structures
* Choosing a reasonable chunk size and overlap
* Maintaining page metadata during chunking
* Generating and storing embeddings efficiently
* Selecting relevant chunks for each question
* Preventing the LLM from answering outside the retrieved context
* Keeping the Streamlit interface simple while exposing useful RAG controls
* Testing retrieval quality with questions that require different levels of reasoning

These areas helped me understand how the individual components of a RAG system work together rather than treating RAG as just an LLM prompt.

---

## Future Improvements

Some improvements I would like to explore:

* Hybrid keyword + vector search
* Reranking retrieved chunks
* Better PDF table extraction
* OCR support for scanned PDFs
* Streaming LLM responses
* More advanced conversation memory
* Document summarization
* Retrieval evaluation metrics
* Support for additional embedding models
* Persistent vector databases for larger document collections

---

## Key Takeaway

This project was built to understand the complete flow of a practical RAG system:

```text
Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector Search
   ↓
Context Retrieval
   ↓
LLM
   ↓
Grounded Answer
```

The main idea is simple: **retrieve the right information first, then let the LLM generate the answer from that information.**

---

## Author

**Thumma Sai Krishna**

Computer Engineering | AI & Software Development

<p>
  <a href="https://github.com/YOUR_USERNAME">GitHub</a>
  •
  <a href="https://www.linkedin.com/in/YOUR_USERNAME/">LinkedIn</a>
</p>

---

## License

This project is intended for learning and educational purposes.
