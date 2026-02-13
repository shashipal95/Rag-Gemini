"""
Simple RAG API with Gemini and Pinecone
FastAPI application with file upload and query endpoints
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import os
from pinecone import Pinecone, ServerlessSpec
import google.genai as genai
from google.genai import types
from PyPDF2 import PdfReader
import docx
import io
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="RAG API with Gemini",
    description="Upload documents and query them using Gemini AI",
    version="1.0.0"
)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_NAME = "gemini-rag"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

print(f"Pinecone API Key loaded: {PINECONE_API_KEY[:10]}...")
print(f"Gemini API Key loaded: {GEMINI_API_KEY[:10]}...")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create or connect to index
def setup_index():
    """Setup Pinecone index"""
    try:
        existing_indexes = [idx['name'] for idx in pc.list_indexes()]
        print(f"Existing indexes: {existing_indexes}")
    except Exception as e:
        print(f"Error listing indexes: {e}")
        raise
    
    if INDEX_NAME not in existing_indexes:
        print(f"Creating new index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # all-MiniLM-L6-v2 dimension
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    else:
        print(f"Using existing index: {INDEX_NAME}")
    
    return pc.Index(INDEX_NAME)

index = setup_index()

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_added: int

# Helper functions
def extract_text_from_file(file: UploadFile) -> str:
    """Extract text from different file types"""
    content = file.file.read()
    
    if file.filename.endswith('.txt'):
        return content.decode('utf-8')
    
    elif file.filename.endswith('.pdf'):
        pdf_reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    elif file.filename.endswith('.docx'):
        doc = docx.Document(io.BytesIO(content))
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .txt, .pdf, or .docx")

def chunk_text(text: str) -> List[str]:
    """Split text into chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    
    return chunks

def get_embedding(text: str) -> List[float]:
    """Generate embedding using Sentence Transformers"""
    try:
        embedding = embedding_model.encode(text)
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

def get_query_embedding(text: str) -> List[float]:
    """Generate embedding for query using Sentence Transformers"""
    try:
        embedding = embedding_model.encode(text)
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        raise

# API Endpoints
@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "RAG API with Gemini",
        "endpoints": {
            "docs": "/docs",
            "upload": "/upload",
            "query": "/query",
            "health": "/health"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        stats = index.describe_index_stats()
        return {
            "status": "healthy",
            "pinecone_connected": True,
            "total_vectors": stats.total_vector_count
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and process a document
    
    - **file**: Upload a .txt, .pdf, or .docx file
    - Returns: Confirmation with number of chunks processed
    """
    try:
        # Extract text from file
        text = extract_text_from_file(file)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="File is empty or unreadable")
        
        # Chunk the text
        chunks = chunk_text(text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid chunks created from file")
        
        print(f"Processing {len(chunks)} chunks from {file.filename}")
        
        # Create embeddings and store in Pinecone
        vectors = []
        for i, chunk in enumerate(chunks):
            print(f"Generating embedding for chunk {i+1}/{len(chunks)}")
            embedding = get_embedding(chunk)
            
            vector_id = f"{file.filename}_{i}"
            metadata = {
                'text': chunk,
                'filename': file.filename,
                'chunk_index': i
            }
            
            vectors.append({
                'id': vector_id,
                'values': embedding,
                'metadata': metadata
            })
        
        # Upsert to Pinecone in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            index.upsert(vectors=batch)
            print(f"Uploaded batch {i//batch_size + 1}")
        
        return UploadResponse(
            message="File uploaded and processed successfully",
            filename=file.filename,
            chunks_added=len(chunks)
        )
    
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the document database
    
    - **question**: Your question
    - **top_k**: Number of relevant chunks to retrieve (default: 3)
    - Returns: AI-generated answer with sources
    """
    try:
        # Get query embedding
        query_embedding = get_query_embedding(request.question)
        
        # Search Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=request.top_k,
            include_metadata=True
        )
        
        if not results['matches']:
            return QueryResponse(
                answer="No relevant information found in the database.",
                sources=[]
            )
        
        # Prepare context from retrieved chunks
        context_parts = []
        sources = []
        
        for i, match in enumerate(results['matches']):
            context_parts.append(f"[{i+1}] {match['metadata']['text']}")
            sources.append({
                'chunk_id': match['id'],
                'filename': match['metadata']['filename'],
                'score': float(match['score']),
                'text': match['metadata']['text'][:200] + "..."
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate answer using Gemini
        prompt = f"""Based on the following context, answer the question. 
If the answer cannot be found in the context, say so clearly.

Context:
{context}

Question: {request.question}

Answer:"""
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        answer = response.text
        
        return QueryResponse(
            answer=answer,
            sources=sources
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.delete("/clear")
async def clear_database():
    """
    Clear all vectors from the database
    
    - Returns: Confirmation message
    """
    try:
        # Delete all vectors
        index.delete(delete_all=True)
        return {"message": "Database cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")

@app.get("/stats")
async def get_stats():
    """
    Get database statistics
    
    - Returns: Current database stats
    """
    try:
        stats = index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)