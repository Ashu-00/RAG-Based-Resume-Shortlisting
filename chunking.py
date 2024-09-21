import re
import os
import fitz
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# Initialize SentenceTransformer model for embedding generation
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize FAISS index
dimension = 384  # Embedding size for SentenceTransformer model
faiss_index = faiss.IndexFlatL2(dimension)

# Number of clusters for KMeans
num_clusters = 8

documents = []
document_embeddings = []
bm25_corpus = []  # This will store the cleaned text for BM25 scoring


def clean_text(text):
    """Cleans and tokenizes the input text by splitting into sentences."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    sentences = re.split(r'(?<=[.!?]) +|\n+', text)  # Split by punctuation or newlines
    return [s.strip() for s in sentences if s]  # Return list of cleaned sentences


def extract_text_from_pdf(pdf_path):
    """Extracts text from PDF using PyMuPDF (fitz)."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        full_text += page.get_text("text")
    return full_text


def bm25_preprocess(text):
    """Preprocess text for BM25 by removing special characters and splitting by spaces."""
    return re.sub(r'\W+', ' ', text.lower()).split()


def add_document_to_index(doc_id: str, text: str):
    """Add document to FAISS and BM25 index after chunking and embedding generation."""
    # Clean and tokenize the document
    sentences = clean_text(text)
    print(f"Number of sentences in doc {doc_id}: {len(sentences)}")

    # Generate sentence embeddings
    sentence_embeddings = model.encode(sentences)

    # Perform KMeans clustering on sentence embeddings
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(sentence_embeddings)

    clusters = {}
    for i, label in enumerate(kmeans.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(sentences[i])

    # Combine sentences in each cluster to form chunks
    for label, chunk_sentences in clusters.items():
        chunk_text = " ".join(chunk_sentences)  # Combine sentences in each cluster
        chunk_embedding = model.encode([chunk_text])[0]  # Generate embedding for chunk
        
        # Add the chunk embedding and text to FAISS and BM25 index
        document_embeddings.append(chunk_embedding)
        documents.append({"id": f"{doc_id}_cluster_{label}", "text": chunk_text})

        # Add chunk embedding to FAISS index
        faiss_index.add(np.array([chunk_embedding], dtype=np.float32))

        # Add chunk text to BM25 corpus
        bm25_corpus.append(bm25_preprocess(chunk_text))


def process_pdf_directory(directory_path: str):
    """Processes all PDFs in a directory."""
    pdf_files = [f for f in os.listdir(directory_path) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory_path, pdf_file)
        print(f"Extracting text for PDF {pdf_file}")
        raw_text = extract_text_from_pdf(pdf_path)
        print("Adding document to vector DB")
        add_document_to_index(pdf_file, raw_text)

    # Initialize BM25 on the full corpus after all documents are processed
    global bm25
    bm25 = BM25Okapi(bm25_corpus)


def retrieve_documents(query: str, top_k: int = 5):
    """Performs FAISS-based semantic search for a query."""
    query_embedding = model.encode([query])[0].reshape(1, -1)
    D, I = faiss_index.search(query_embedding, top_k)
    retrieved_docs = [documents[i] for i in I[0]]  # Retrieve most similar documents
    return retrieved_docs


def hybrid_search(query: str, top_k=5):
    """Performs hybrid search using both BM25 and FAISS."""
    # Preprocess the query for BM25
    query_tokens = bm25_preprocess(query)

    # BM25 Search: Retrieve top-k documents using BM25
    bm25_scores = bm25.get_scores(query_tokens)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]

    # FAISS Search: Convert the query into an embedding and search FAISS
    query_embedding = model.encode([query])[0].reshape(1, -1)
    faiss_distances, faiss_indices = faiss_index.search(query_embedding, top_k)

    # Normalize FAISS distances to be comparable with BM25 scores (optional)
    faiss_scores = 1 / (1 + faiss_distances.flatten())

    # Combine FAISS and BM25 results
    hybrid_scores = {}

    # Assign BM25 scores to hybrid scores
    for i in top_bm25_indices:
        hybrid_scores[i] = hybrid_scores.get(i, 0) + bm25_scores[i]

    # Assign FAISS scores to hybrid scores
    for i, score in zip(faiss_indices.flatten(), faiss_scores):
        hybrid_scores[i] = hybrid_scores.get(i, 0) + score

    # Sort documents by combined score
    sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # Return the top documents
    return [(documents[idx]["id"], documents[idx]["text"], hybrid_scores[idx]) for idx, score in sorted_results]


if __name__ == "__main__":
    # Process the PDF directory and build indices
    process_pdf_directory("docs")

    # Input query
    q = input("Enter your query: ")
    print(f"Retrieving documents based on query '{q}'")
    
    # Perform hybrid search and print results
    print("Hybrid Search Results")
    results = hybrid_search(q, top_k=5)
    for i, (id, doc, score) in enumerate(results):
        print(f"\nResult {i + 1} (Score: {score:.2f}) ID - {id} :\n{doc}")
    
    
    print("\nNormal FAISS search results")
    normal_results = retrieve_documents(q)
    for i in normal_results:
        print(f"Id {i['id']} :\n{i['text']}", end="\n\n")
