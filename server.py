from chunking import process_pdf_directory,hybrid_search
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
import groq
from fastapi import FastAPI, HTTPException

class Query(BaseModel):
    text: str

class Response(BaseModel):
    answer: str
    sources: List[str]


def preprocess_documents(directory_path: str):
    """
    Preprocess documents from a given directory.
    """
    process_pdf_directory(directory_path)
    
def rag_pipeline(query: str, top_k: int = 5):
    """
    Implement the Retrieval-Augmented Generation pipeline.
    """
    # Retrieval step
    retrieved_docs = hybrid_search(query, top_k)
    
    # Prepare context for the language model
    context = "\n\n".join([f" Name: {doc[0].split('.')[0]} Excerpt- {doc[1]}" for i, doc in enumerate(retrieved_docs)])
    #print(context)
    
    # Generate response using Groq
    prompt = f"Based Solely on the following context, answer the query: '{query}'\n\nContext:\n{context}\n\nAnswer:"
    
    completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You would be given resume excerpts with names. Your final answer must be who would be a good fit for the given role."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.1-8b-instant",
        max_tokens=1024,
        temperature=0.5,
    )
    
    answer = completion.choices[0].message.content
    sources = [doc[0] for doc in retrieved_docs]  # Document IDs
    
    return answer, sources

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")
groq_client = groq.Client(api_key=groq_api_key)

@app.post("/query", response_model=Response)
async def query_endpoint(query: Query):
    try:
        answer, sources = rag_pipeline(query.text)
        return Response(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Startup event to preprocess documents
@app.on_event("startup")
async def startup_event():
    # Preprocess documents in the "docs" directory
    preprocess_documents("docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 