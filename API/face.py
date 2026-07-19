
# ============================================================================
# FACE RECOGNITION and verification APIs
# ============================================================================
import traceback
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from drf_spectacular.utils import extend_schema
from django.contrib.auth import get_user_model
from django.conf import settings
from .facehelper import _run_face_verification, _read_uploaded_images
from chat.model_singleton import get_processor
from Profile.models import Record
from chat.utils import delete_face_embedding
from ChatApp.ratelimiting import EnrollmentPerUserThrottle, EnrollmentPerIPThrottle, VerificationPerUserThrottle
from ChatApp.throttling import ThrottledAPIView
from Profile.models import Record
User = get_user_model()


@extend_schema(operation_id='face_enrollment', tags=['FaceRecognition'])
class FaceEnrollmentAPIView(ThrottledAPIView):
    """
    POST: Train face embeddings from images
    """

    permission_classes = [IsAuthenticated]
    parser_classes = (MultiPartParser, FormParser)
    throttle_classes = [EnrollmentPerUserThrottle, EnrollmentPerIPThrottle]

    def post(self, request):
        """Train face embeddings"""
        try:
            user_id = request.POST.get("user_id", str(request.user.id))
            images = request.FILES.getlist("images")
            if not images:
                return Response(
                    {"status": "error", "message": "No images provided", "samples": 0},
                    status=status.HTTP_400_BAD_REQUEST,
                )
                
            if len(images) != 3:
                return Response(
                    {
                        "status": "error",
                        "message": f"Exactly 3 images required, got {len(images)}",
                        "samples": 0,
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )
            
            if Record.objects.filter(user_id=user_id, image_added="1").exists():
                return Response(
                    {
                        "status": "error",
                        "message": "Face embeddings already exist for this user",
                        "samples": 0,
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )            
            image_bytes_list = _read_uploaded_images(images)
            distance_threshold = getattr(settings, "FACE_DISTANCE", 0.45)

            processor = get_processor()
            result = processor.train_user_embeddings(
                user_id, image_bytes_list, distance_threshold=distance_threshold
            )
            processor.cleanup()

            if result.get("status") == "success":
                try:
                    user = User.objects.get(id=user_id)
                    record, _ = Record.objects.get_or_create(user=user)
                    record.image_added = "1"
                    record.save()
                except Record.DoesNotExist:
                    pass

                return Response(result, status=status.HTTP_200_OK)
            
            return Response(result,status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            traceback.print_exc()
            return Response(
                {"status": "error", "message": str(e), "samples": 0},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


@extend_schema(operation_id='face_verification', tags=['FaceRecognition'])
class FaceVerificationAPIView(ThrottledAPIView):
    """
    POST: Verify user identity from face images
    """

    permission_classes = [IsAuthenticated]
    parser_classes = (MultiPartParser, FormParser)
    throttle_classes = [VerificationPerUserThrottle]
    def post(self, request):
        """Verify face identity"""
        try:
            user_id = request.POST.get("user_id")
            images = request.FILES.getlist("images")
            threshold = float(request.POST.get("threshold", 80)) / 100
            image_path = request.POST.get("image_path")  # <-- added
            message_id = request.POST.get("message_id")

            if not user_id:
                return Response(
                    {
                        "status": "error",
                        "message": "user_id is required",
                        "verified": False,
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if not image_path and not message_id:  # <-- added
                return Response(
                    {
                        "status": "error",
                        "message": "image_path or message_id is required",
                        "verified": False,
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if not images:
                return Response(
                    {
                        "status": "error",
                        "message": "No images provided",
                        "verified": False,
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )
                
            if len(images) != 1:
                return Response(
                    {
                        "status": "error",
                        "message": f"Exactly 1 image required, got {len(images)}",
                        "verified": False,
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )
            
            if Record.objects.filter(user_id=user_id, image_added="0").exists():
                return Response(
                    {
                        "status": "error",
                        "message": "No face embeddings found for this user - please enroll first",
                        "verified": False,
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )
            result = _run_face_verification(
                user_id, _read_uploaded_images(images), threshold
            )

            # Store image_path in session only if face verification passed
            if result.get("verified") is True:  # <-- added
                if image_path:
                    request.session["face_verified_for"] = image_path
                if message_id:
                    request.session[f"face_verified_message_{message_id}"] = True
                request.session.modified = True

                return Response(result, status=status.HTTP_200_OK)
            
            return Response(result, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            traceback.print_exc()
            return Response(
                {"status": "error", "message": str(e), "verified": False},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


@extend_schema(operation_id='delete_face', tags=['FaceRecognition'])
class DeleteFaceAPIView(APIView):
    """
    POST: Delete user's face embeddings
    """

    permission_classes = [IsAuthenticated]

    def post(self, request, user_id):
        """Delete face embeddings for user"""
        if str(request.user.id) != str(user_id):
            return Response(
                {"status": "error", "message": "Unauthorized"},
                status=status.HTTP_403_FORBIDDEN,
            )

        try:
            password = request.data.get("password")
            if not password or not request.user.check_password(password):
                return Response(
                    {"status": "error", "message": "Wrong password"},
                    status=status.HTTP_403_FORBIDDEN,
                )

            delete_face_embedding(user_id)
            return Response(
                {"status": "success", "message": "Face deleted successfully"},
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            return Response(
                {"status": "error", "message": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )