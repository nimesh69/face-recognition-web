# faiss_manager.py
import os
import faiss
import numpy as np
import pickle
import json


class FAISSEmbeddingManager:
    def __init__(self, embeddings_dir, dimension=512):
        """
        Initialize FAISS index manager.

        Args:
            embeddings_dir: Directory to store FAISS index and metadata
            dimension: Embedding dimension (512 for buffalo_m)
        """
        self.embeddings_dir = embeddings_dir
        self.dimension = dimension
        self.index_path = os.path.join(embeddings_dir, "faiss_index.bin")
        self.metadata_path = os.path.join(embeddings_dir, "metadata.pkl")

        os.makedirs(embeddings_dir, exist_ok=True)

        # Initialize or load index
        if os.path.exists(self.index_path):
            self.load_index()
        else:
            self.create_new_index()

    def create_new_index(self):
        """Create a new FAISS index."""
        # Using L2 distance (can also use IndexFlatIP for cosine similarity)
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
        self.user_ids = []  # Maps index position to user_id
        self.user_metadata = {}  # Stores additional user info
        print("Created new FAISS index")

    def load_index(self):
        """Load existing FAISS index and metadata."""
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                data = pickle.load(f)
                self.user_ids = data["user_ids"]
                self.user_metadata = data.get("metadata", {})

            # Migration: Convert IndexFlat to IndexIDMap if needed
            if not isinstance(self.index, faiss.IndexIDMap):
                print("Migrating FAISS index to IndexIDMap...")
                new_index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
                if self.index.ntotal > 0:
                    # Reconstruct all vectors
                    vectors = np.zeros(
                        (self.index.ntotal, self.dimension), dtype="float32"
                    )
                    for i in range(self.index.ntotal):
                        vectors[i] = self.index.reconstruct(i)

                    # IDs from user_ids list
                    ids = np.array([int(uid) for uid in self.user_ids], dtype=np.int64)
                    new_index.add_with_ids(vectors, ids)
                self.index = new_index
                self.save_index()

            print(f"Loaded FAISS index with {self.index.ntotal} embeddings")
        except Exception as e:
            print(f"Error loading index: {e}")
            self.create_new_index()

    def save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "wb") as f:
                pickle.dump(
                    {"user_ids": self.user_ids, "metadata": self.user_metadata}, f
                )
            print(f"Saved FAISS index with {self.index.ntotal} embeddings")
        except Exception as e:
            print(f"Error saving index: {e}")

    def add_user_embedding(self, user_id, embedding, metadata=None):
        """
        Add or update a user's embedding.

        Args:
            user_id: User identifier (string or int)
            embedding: Face embedding vector (512-d numpy array)
            metadata: Optional additional user info (dict)
        """
        user_id = str(user_id)

        # Normalize embedding for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        embedding = embedding.reshape(1, -1).astype("float32")

        # Check if user already exists
        if user_id in self.user_ids:
            self.remove_user_embedding(user_id)

        # Add new embedding
        self.index.add_with_ids(embedding, np.array([int(user_id)], dtype=np.int64))
        self.user_ids.append(user_id)

        if metadata:
            self.user_metadata[user_id] = metadata

        self.save_index()
        print(f"Added/Updated embedding for user {user_id}")

    def remove_user_embedding(self, user_id):
        """Remove a user's embedding from the index."""
        user_id = str(user_id)

        if user_id not in self.user_ids:
            print(f"User {user_id} not found in index")
            return

        # FAISS IndexIDMap supports remove_ids
        self.index.remove_ids(np.array([int(user_id)], dtype=np.int64))

        if user_id in self.user_ids:
            self.user_ids.remove(user_id)
        if user_id in self.user_metadata:
            del self.user_metadata[user_id]

        self.save_index()
        print(f"Removed embedding for user {user_id}")

    def search(self, query_embedding, k=1, threshold=None):
        """
        Search for similar embeddings.

        Args:
            query_embedding: Query face embedding (512-d numpy array)
            k: Number of nearest neighbors to return
            threshold: Optional distance threshold (L2 distance)

        Returns:
            List of tuples: [(user_id, distance), ...]
        """
        if self.index.ntotal == 0:
            return []

        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype("float32")

        # Search
        distances, indices = self.index.search(
            query_embedding, min(k, self.index.ntotal)
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # Valid index found
                if threshold is None or dist <= threshold:
                    user_id = str(idx)  # With IndexIDMap, idx is the user_id
                    results.append((user_id, float(dist)))

        return results

    def get_user_count(self):
        """Get total number of users in index."""
        return self.index.ntotal

    def user_exists(self, user_id):
        """Check if a user exists in the index."""
        return str(user_id) in self.user_ids

    def get_all_user_ids(self):
        """Get list of all user IDs."""
        return self.user_ids.copy()
