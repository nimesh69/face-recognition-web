"""
Server-side face embedding processor for handling images received from client.
Processes images and manages FAISS embeddings without camera dependency.
"""


import os
import sys
import json
import time
import gc
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image
import cv2
import io
from typing import List, Tuple, Dict
from .faiss_manager import FAISSEmbeddingManager
from .anti_spoofing import FaceAntiSpoofing


# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)



class FaceEmbeddingProcessor:
    """Process face images and manage embeddings using InsightFace + FAISS"""


    def __init__(self):
        """Initialize the face analyzer and FAISS manager"""
        models_dir = os.path.join(PROJECT_ROOT, "models")
        self.app = FaceAnalysis(name="buffalo_m", root=models_dir)
        self.app.prepare(ctx_id=-1)


        embeddings_path = os.path.join(PROJECT_ROOT, "embeddings")


        # Initialize Anti-Spoofing
        model_path = os.path.join(PROJECT_ROOT, "models/anti_spoof", "best_model.onnx")
        self.spoof_detector = FaceAntiSpoofing(model_path)

        self._load_faiss_index()

    def _load_faiss_index(self):
        embeddings_path = os.path.join(PROJECT_ROOT, "embeddings")
        self.faiss_manager = FAISSEmbeddingManager(embeddings_path, dimension=512)

    def reload_faiss(self):
        """Call this explicitly after registration, not on every verify."""
        self._load_faiss_index()

    def process_image_bytes(self, image_bytes: bytes) -> Tuple[np.ndarray, Dict]:
        """
        Process image from bytes and extract face embedding.


        Args:
            image_bytes: Image data as bytes


        Returns:
            Tuple of (embedding, metadata) or (None, error_dict) if no face found
        """
        try:
            # Convert bytes to OpenCV format
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


            if frame is None:
                return None, {"error": "Invalid image format"}


            # Detect faces
            faces = self.app.get(frame)


            if len(faces) == 0:
                return None, {
                    "error": "No face detected in image",
                    "error_code": "no_face",
                }


            if len(faces) > 1:
                return None, {
                    "error": f"Multiple faces detected ({len(faces)}). Only one face per image allowed.",
                    "error_code": "multiple_faces",
                }


            # Get embedding from first (only) face
            face = faces[0]


            if face.det_score < 0.60:
                return None, {
                    "error": f"Fake face detected: Low detection confidence ({face.det_score:.2f})",
                    "error_code": "low_det_score",
                }


            # Check for Liveness / Anti-Spoofing
            is_real, score, msg = self.spoof_detector.check(frame, face.bbox)
            if not is_real:
                return None, {
                    "error": f"Fake face detected: {msg}",
                    "error_code": "spoof",
                    "liveness_score": float(score),
                }


            embedding = face.embedding


            # Get face bounding box for metadata
            bbox = face.bbox.astype(int)


            return embedding, {
                "success": True,
                "bbox": bbox.tolist(),
                "confidence": float(face.det_score),
                "liveness_score": float(score),
            }


        except Exception as e:
            return None, {"error": str(e)}


    def train_user_embeddings(
        self,
        user_id: str,
        image_bytes_list: List[bytes],
        distance_threshold: float = 0.45,
    ) -> Dict:
        """
        Process multiple training images and save combined embedding to FAISS.


        Args:
            user_id: User identifier
            image_bytes_list: List of image bytes for training
            distance_threshold: Threshold for checking duplicates


        Returns:
            Dictionary with status, samples captured, and any errors
        """
        # print(f"\n[FAISS] ========== STARTING TRAINING ==========")
        # print(f"[FAISS] User ID: '{user_id}' (type: {type(user_id)})")
        # print(f"[FAISS] Total images to process: {len(image_bytes_list)}")


        all_embeddings = []
        liveness_scores = []
        failed_images = []
        spoof_count = 0


        for idx, image_bytes in enumerate(image_bytes_list):
            embedding, metadata = self.process_image_bytes(image_bytes)


            if embedding is not None:
                all_embeddings.append(embedding)
                liveness_scores.append(metadata.get("liveness_score", 0.0))
                print(
                    f"[FAISS] Image {idx+1}/{len(image_bytes_list)}: Face detected ✓ (confidence: {metadata.get('confidence', 'N/A')})"
                )
            else:
                failed_images.append(
                    {"index": idx, "error": metadata.get("error", "Unknown error")}
                )
                print(
                    f"[FAISS] Image {idx+1}/{len(image_bytes_list)}: {metadata.get('error', 'Unknown error')} ✗"
                )


                error_code = metadata.get("error_code")
                if error_code in ("no_face", "multiple_faces", "low_det_score"):
                    print(
                        f"[FAISS] ✗ ERROR: Rejecting training early - {metadata.get('error', 'Unknown error')}"
                    )
                    return {
                        "status": "error",
                        "message": metadata.get("error", "Image validation failed"),
                        "samples": 0,
                        "failed_images": failed_images,
                    }


                if error_code == "spoof":
                    spoof_count += 1
                    if spoof_count >= 2:
                        print(
                            f"[FAISS] ✗ ERROR: Rejecting training early - {spoof_count} spoof images detected"
                        )
                        return {
                            "status": "error",
                            "message": "Liveness check failed. 2 spoof images detected.",
                            "samples": 0,
                            "failed_images": failed_images,
                        }


        if len(all_embeddings) == 0:
            print(f"[FAISS] ✗ ERROR: No valid faces found in any image!")
            print(f"[FAISS] =================================")
            return {
                "status": "error",
                "message": "No valid faces found in any image",
                "samples": 0,
                "failed_images": failed_images,
            }


        # Check average liveness score
        print(f"[FAISS] Checking average liveness score...{liveness_scores}")
        avg_liveness = np.mean(liveness_scores) if liveness_scores else 0.0
        print(f"[FAISS] Average liveness score: {avg_liveness:.4f}")


        if avg_liveness < 0.7:
            print(
                f"[FAISS] ✗ ERROR: Average liveness score too low ({avg_liveness:.4f} < 0.7)"
            )
            return {
                "status": "error",
                "message": f"Liveness check failed. Average score: {avg_liveness:.4f}",
                "samples": 0,
                "failed_images": failed_images,
            }


        print(
            f"\n[FAISS] Valid faces found: {len(all_embeddings)}/{len(image_bytes_list)}"
        )
        print(f"[FAISS] Averaging {len(all_embeddings)} embeddings...")


        # Combine embeddings into single averaged embedding
        combined_embedding = np.mean(all_embeddings, axis=0)
        combined_embedding /= np.linalg.norm(combined_embedding)


        print(f"[FAISS] Combined embedding shape: {combined_embedding.shape}")
        print(f"[FAISS] Embedding norm: {np.linalg.norm(combined_embedding):.4f}")


        # Check for existing face
        print(
            f"[FAISS] Checking for existing face with threshold {distance_threshold}..."
        )
        existing_matches = self.faiss_manager.search(
            combined_embedding, k=1, threshold=distance_threshold
        )


        if existing_matches:
            found_id, dist = existing_matches[0]
            # If the found face is NOT the current user (i.e., it's someone else)
            if str(found_id) != str(user_id):
                print(
                    f"[FAISS] ✗ ERROR: Similar face found for user {found_id} (distance: {dist:.4f})"
                )
                return {
                    "status": "error",
                    "message": f"Similar face found in database. Please remove existing face",
                    "samples": 0,
                    "failed_images": failed_images,
                }


        print(f"[FAISS] Saving embedding to FAISS for user '{user_id}'...")


        # Save to FAISS
        self.faiss_manager.add_user_embedding(
            user_id=user_id,
            embedding=combined_embedding,
            metadata={
                "total_samples": len(all_embeddings),
                "timestamp": time.time(),
                "failed_images": len(failed_images),
            },
        )


        # Verify it was saved
        if self.faiss_manager.user_exists(user_id):
            print(f"[FAISS] ✓ Successfully saved to FAISS")
        else:
            print(f"[FAISS] ✗ WARNING: User not found in FAISS after saving!")


        total_users = self.faiss_manager.get_user_count()
        all_users = self.faiss_manager.get_all_user_ids()
        print(f"[FAISS] Total users in FAISS: {total_users}")
        print(f"[FAISS] All user IDs in DB: {all_users}")


        print(f"[FAISS] ========== TRAINING COMPLETE ==========\n")


        # Save to FAISS
        self.faiss_manager.add_user_embedding(
            user_id=user_id,
            embedding=combined_embedding,
            metadata={
                "total_samples": len(all_embeddings),
                "timestamp": time.time(),
                "failed_images": len(failed_images),
            },
        )


        print(f"[FAISS] Successfully trained user {user_id}")


        result = {
            "status": "success",
            "message": f"Embedding trained successfully from {len(all_embeddings)} images",
            "user_id": user_id,
            "samples": len(all_embeddings),
            "failed_images": len(failed_images),
            "failed_details": failed_images if failed_images else None,
            "total_users_in_db": self.faiss_manager.get_user_count(),
        }


        return result


    def verify_user_face(
        self,
        target_user_id: str,
        verification_images: List[bytes],
        required_match_percentage: float = 0.80,
        distance_threshold: float = 0.45,
    ) -> Dict:
        """
        Verify if verification images match the target user.


        Args:
            target_user_id: User to verify against
            verification_images: List of verification image bytes
            required_match_percentage: Required match percentage (0-1)
            distance_threshold: L2 distance threshold for matching


        Returns:
            Dictionary with verification result and match details
        """


        print(f"\n[VERIFY] Starting verification for user {target_user_id}")
        print(
            f"[VERIFY] Target user ID type: {type(target_user_id)}, value: '{target_user_id}'"
        )
        print(f"[VERIFY] Distance threshold: {distance_threshold}")
        print(f"[VERIFY] Required match percentage: {required_match_percentage * 100}%")


        if not self.faiss_manager.user_exists(target_user_id):
            print(f"[VERIFY] ERROR: User {target_user_id} not found in FAISS database")
            print(
                f"[VERIFY] Available users in DB: {self.faiss_manager.get_all_user_ids()}"
            )
            return {
                "status": "error",
                "message": f"User {target_user_id} not found in database",
                "match_percentage": 0,
                "verified": False,
            }


        print(f"[VERIFY] User {target_user_id} found in database ✓")


        match_count = 0
        processed_count = 0
        failed_images = []
        liveness_scores = []
        match_details = []
        required_matches = 2 if len(verification_images) >= 3 else 1


        for idx, image_bytes in enumerate(verification_images):
            print(
                f"\n[VERIFY] Processing verification image {idx + 1}/{len(verification_images)}..."
            )
            embedding, metadata = self.process_image_bytes(image_bytes)
            processed_count += 1


            if embedding is None:
                print(
                    f"[VERIFY] Image {idx + 1}: FAILED - {metadata.get('error', 'Unknown error')}"
                )
                failed_images.append(
                    {"index": idx, "error": metadata.get("error", "Unknown error")}
                )
                if metadata.get("error_code") in (
                    "no_face",
                    "multiple_faces",
                    "low_det_score",
                    "spoof",
                ):
                    print(
                        f"[VERIFY] ✗ ERROR: Rejecting verification early - {metadata.get('error', 'Unknown error')}"
                    )
                    return {
                        "status": "error",
                        "message": metadata.get("error", "Image validation failed"),
                        "match_percentage": 0,
                        "verified": False,
                        "processed_images": processed_count,
                        "matched_images": match_count,
                        "failed_images": len(failed_images),
                        "failed_details": failed_images,
                        "match_details": match_details,
                    }


                continue


            liveness_scores.append(metadata.get("liveness_score", 0.0))
            print(
                f"[VERIFY] Image {idx + 1}: Face detected ✓, confidence: {metadata.get('confidence', 'N/A')}"
            )


            # Search in FAISS
            print(f"[VERIFY] Searching FAISS for matching user...")
            results = self.faiss_manager.search(
                embedding, k=1, threshold=distance_threshold
            )


            print(f"[VERIFY] FAISS search results: {results}")


            if results:
                detected_id, distance = results[0]
                detected_id_str = str(detected_id).strip()
                target_id_str = str(target_user_id).strip()


                print(
                    f"[VERIFY] Detected user ID: '{detected_id_str}' (type: {type(detected_id)})"
                )
                print(
                    f"[VERIFY] Target user ID: '{target_id_str}' (type: {type(target_user_id)})"
                )
                print(f"[VERIFY] Distance: {distance}")
                print(f"[VERIFY] Similarity: {max(0, (1 - distance) * 100):.1f}%")


                if detected_id_str == target_id_str:
                    match_count += 1
                    print(f"[VERIFY] ✓ MATCH FOUND!")
                    match_details.append(
                        {
                            "index": idx,
                            "matched": True,
                            "distance": float(distance),
                            "similarity": float(max(0, (1 - distance) * 100)),
                        }
                    )
                    if match_count >= required_matches:
                        print(
                            f"[VERIFY] Required matches reached ({match_count}/{required_matches}). Stopping early."
                        )
                        break
                else:
                    print(
                        f"[VERIFY] ✗ No match - detected different user ({detected_id_str} != {target_id_str})"
                    )
                    match_details.append(
                        {
                            "index": idx,
                            "matched": False,
                            "detected_user": detected_id_str,
                            "distance": float(distance),
                            "similarity": float(max(0, (1 - distance) * 100)),
                        }
                    )
            else:
                print(
                    f"[VERIFY] ✗ No match - search returned no results above threshold"
                )
                match_details.append(
                    {
                        "index": idx,
                        "matched": False,
                        "error": "No match found above threshold",
                    }
                )


        # Check average liveness score
        print(f"[VERIFY] Checking average liveness score...{liveness_scores}")
        avg_liveness = np.mean(liveness_scores) if liveness_scores else 0.0
        print(f"[VERIFY] Average liveness score: {avg_liveness:.4f}")


        if avg_liveness < 0.7:
            print(f"[VERIFY] ✗ Liveness check failed")
            return {
                "status": "error",
                "message": f"Liveness check failed. Average score: {avg_liveness:.2f}",
                "match_percentage": 0,
                "verified": False,
            }


        # Calculate match percentage
        match_percentage = match_count / processed_count if processed_count > 0 else 0
        verified = match_count >= required_matches


        print(f"\n[VERIFY] ========== VERIFICATION SUMMARY ==========")
        print(f"[VERIFY] Total images processed: {processed_count}")
        print(f"[VERIFY] Total images matched: {match_count}")
        print(f"[VERIFY] Failed images: {len(failed_images)}")
        print(f"[VERIFY] Match percentage: {match_percentage * 100:.1f}%")
        print(f"[VERIFY] Required percentage: {required_match_percentage * 100:.1f}%")
        print(f"[VERIFY] VERIFIED: {verified}")
        print(f"[VERIFY] ==========================================\n")


        result = {
            "status": "success",
            "target_user_id": target_user_id,
            "processed_images": processed_count,
            "matched_images": match_count,
            "failed_images": len(failed_images),
            "match_percentage": float(match_percentage * 100),
            "required_percentage": float(required_match_percentage * 100),
            "verified": verified,
            "match_details": match_details,
            "failed_details": failed_images if failed_images else None,
        }


        return result


    def cleanup(self):
        """Clean up resources"""
        gc.collect()
