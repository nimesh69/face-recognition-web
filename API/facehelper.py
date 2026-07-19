from django.conf import settings
 
from chat.model_singleton import get_processor

# ============================================================================
# HELPER FUNCTIONS FOR FACE RECOGNITION
# ============================================================================
def _run_face_verification(user_id, image_bytes_list, threshold):
    """Run face verification and return the result dict."""
    distance_threshold = getattr(settings, "FACE_DISTANCE", 0.45)
    processor = get_processor()
    result = processor.verify_user_face(
        user_id,
        image_bytes_list,
        required_match_percentage=threshold,
        distance_threshold=distance_threshold,
    )
    processor.cleanup()
    return result


def _read_uploaded_images(file_list):
    """Convert a list of uploaded file objects to a list of raw bytes."""
    return [f.read() for f in file_list]