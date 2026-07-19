# server.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import numpy as np
import cv2
import faiss
import insightface
from insightface.app import FaceAnalysis
import pickle
import os
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import shutil
from pathlib import Path

app = FastAPI(title="Face Recognition System")

# Enable CORS for web client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
EMBEDDING_DIM = 512
FAISS_INDEX_PATH = "face_index.faiss"
METADATA_PATH = "face_metadata.pkl"

# Initialize InsightFace
print("🚀 Loading InsightFace model...")
face_analyzer = FaceAnalysis(name='buffalo_l', root='./models')
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
print("✅ Model loaded successfully")

executor = ThreadPoolExecutor(max_workers=4)

def init_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        print(f"📊 Loaded index with {index.ntotal} vectors")
        return index
    else:
        # Using IndexIDMap with IndexFlatIP for cosine similarity and deletion support
        index = faiss.IndexIDMap(faiss.IndexFlatIP(EMBEDDING_DIM))
        print("🆕 Created new FAISS index with IDMap")
        return index

def load_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'rb') as f:
            return pickle.load(f)
    return {}

def save_metadata(metadata):
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)

faiss_index = init_faiss_index()
user_metadata = load_metadata()

class EmbeddingResponse(BaseModel):
    user_id: str
    successful_uploads: int
    failed_uploads: int
    embeddings_stored: int
    details: List[dict]

def extract_embedding(image_path: str):
    """Extract face embedding from image using InsightFace"""
    img = cv2.imread(image_path)
    if img is None:
        return None, "Failed to load image"
    
    faces = face_analyzer.get(img)
    
    if len(faces) == 0:
        return None, "No face detected"
    elif len(faces) > 1:
        return None, "Multiple faces detected"
    
    # Get embedding and normalize
    embedding = faces[0].embedding
    
    # Normalize and reshape as specified
    embedding = embedding / np.linalg.norm(embedding)
    embedding = embedding.reshape(1, -1).astype('float32')
    
    return embedding, None

@app.get("/")
async def root():
    """Serve the main web interface"""
    return FileResponse("static/index.html")

@app.post("/register_faces", response_model=EmbeddingResponse)
async def register_faces(
    user_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Register multiple face images for a user"""
    if len(files) > 30:
        raise HTTPException(status_code=400, detail="Maximum 30 images allowed")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    
    results = []
    embeddings_list = []
    successful = 0
    failed = 0
    
    for idx, file in enumerate(files):
        temp_path = None
        try:
            if not file.content_type or not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": "Invalid file type"
                })
                failed += 1
                continue
            
            temp_path = UPLOAD_DIR / f"{user_id}_{idx}_{datetime.now().timestamp()}.jpg"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Extract embedding (run in thread pool)
            loop = asyncio.get_event_loop()
            embedding, error = await loop.run_in_executor(
                executor, extract_embedding, str(temp_path)
            )
            
            if error:
                results.append({
                    "filename": file.filename,
                    "status": "failed", 
                    "error": error
                })
                failed += 1
            else:
                # embedding is already shaped (1, 512) from extract_embedding
                # Remove the batch dimension for FAISS storage (FAISS expects (N, D) where N is number of vectors)
                embeddings_list.append(embedding[0])  # Take first (and only) row, shape becomes (512,)
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "face_detected": True
                })
                successful += 1
                
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "failed",
                "error": str(e)
            })
            failed += 1
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
    
    # Store embeddings in FAISS
    if embeddings_list:
        # Stack embeddings: list of (512,) arrays -> (N, 512) array
        embeddings_array = np.stack(embeddings_list)
        print(f"Storing embeddings with shape: {embeddings_array.shape}")  # Debug print
        
        # Generate IDs
        if user_metadata:
            start_idx = max(user_metadata.keys()) + 1
        else:
            start_idx = 0
        ids = np.arange(start_idx, start_idx + len(embeddings_list)).astype('int64')
        
        # Add to FAISS with IDs
        faiss_index.add_with_ids(embeddings_array, ids)
        
        # Update metadata
        for i in range(len(embeddings_list)):
            vector_id = int(ids[i])
            user_metadata[vector_id] = {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "filename": results[i]["filename"]
            }
        
        # Persist
        faiss.write_index(faiss_index, FAISS_INDEX_PATH)
        save_metadata(user_metadata)
        
        print(f"Total vectors in index: {faiss_index.ntotal}")  # Debug print
    
    return EmbeddingResponse(
        user_id=user_id,
        successful_uploads=successful,
        failed_uploads=failed,
        embeddings_stored=len(embeddings_list),
        details=results
    )

@app.post("/search_face")
async def search_face(file: UploadFile = File(...), top_k: int = 5):
    """Search for similar faces in the database"""
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    temp_path = UPLOAD_DIR / f"search_{datetime.now().timestamp()}.jpg"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        loop = asyncio.get_event_loop()
        embedding, error = await loop.run_in_executor(
            executor, extract_embedding, str(temp_path)
        )
        
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        # embedding is (1, 512), FAISS search expects same shape
        print(f"Search embedding shape: {embedding.shape}")  # Debug print
        
        # Search
        distances, indices = faiss_index.search(embedding, min(top_k, faiss_index.ntotal))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx in user_metadata:
                results.append({
                    "user_id": user_metadata[idx]["user_id"],
                    "confidence": float(dist),  # Cosine similarity (0-1) since vectors are normalized
                    "timestamp": user_metadata[idx]["timestamp"]
                })
        
        return {
            "matches": results,
            "total_searched": faiss_index.ntotal
        }
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    unique_users = len(set(m["user_id"] for m in user_metadata.values()))
    return {
        "total_vectors": faiss_index.ntotal,
        "total_users": unique_users,
        "index_size_mb": round(os.path.getsize(FAISS_INDEX_PATH) / (1024*1024), 2) if os.path.exists(FAISS_INDEX_PATH) else 0
    }

@app.delete("/user/{user_id}")
async def delete_user(user_id: str):
    """Remove all embeddings for a specific user"""
    global faiss_index, user_metadata
    
    # Find IDs to remove
    ids_to_remove = []
    for idx, meta in user_metadata.items():
        if meta["user_id"] == user_id:
            ids_to_remove.append(idx)
    
    if not ids_to_remove:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Remove from FAISS
    ids_array = np.array(ids_to_remove, dtype='int64')
    faiss_index.remove_ids(ids_array)
    
    # Remove from metadata
    for idx in ids_to_remove:
        del user_metadata[idx]
        
    # Persist
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    save_metadata(user_metadata)
    
    return {
        "status": "success",
        "vectors_removed": len(ids_to_remove),
        "remaining_vectors": faiss_index.ntotal
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)