# chat/anti_spoofing.py
import cv2
import numpy as np
import onnxruntime
import os


class FaceAntiSpoofing:
    def __init__(self, model_path, model_img_size=128, bbox_expansion_factor=1.5):
        self.model_path = model_path
        self.model_img_size = model_img_size
        self.bbox_expansion_factor = bbox_expansion_factor
        self.session = None

        if os.path.exists(model_path):
            try:
                providers = ["CPUExecutionProvider"]
                if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                    providers = ["CUDAExecutionProvider"] + providers
                self.session = onnxruntime.InferenceSession(model_path, providers=providers)
                self.input_name = self.session.get_inputs()[0].name
            except Exception as e:
                print(f"[FAS] Error loading model: {e}")
        else:
            print(f"[FAS] Warning: Model not found at {model_path}. Anti-spoofing checks will be skipped.")

    def _crop(self, img, bbox):
        """Exact replica of demo's crop() from src/inference/preprocess.py"""
        original_height, original_width = img.shape[:2]
        x1, y1, x2, y2 = bbox.astype(float)

        # bbox is (x1,y1,x2,y2) — convert to x,y,w,h style expected by crop()
        x, y, w, h = x1, y1, x2, y2
        w = w - x
        h = h - y

        if w <= 0 or h <= 0:
            raise ValueError("Invalid bbox dimensions")

        max_dim = max(w, h)
        center_x = x + w / 2
        center_y = y + h / 2

        scale = self.bbox_expansion_factor
        x = int(center_x - max_dim * scale / 2)
        y = int(center_y - max_dim * scale / 2)
        crop_size = int(max_dim * scale)

        crop_x1 = max(0, x)
        crop_y1 = max(0, y)
        crop_x2 = min(original_width,  x + crop_size)
        crop_y2 = min(original_height, y + crop_size)

        top_pad    = int(max(0, -y))
        left_pad   = int(max(0, -x))
        bottom_pad = int(max(0, (y + crop_size) - original_height))
        right_pad  = int(max(0, (x + crop_size) - original_width))

        cropped = img[crop_y1:crop_y2, crop_x1:crop_x2, :]

        result = cv2.copyMakeBorder(
            cropped, top_pad, bottom_pad, left_pad, right_pad,
            cv2.BORDER_REFLECT_101
        )
        return result

    def _preprocess(self, img):
        """Exact replica of demo's preprocess() — letterbox + normalize to [0,1]"""
        new_size = self.model_img_size
        old_size = img.shape[:2]
        ratio = float(new_size) / max(old_size)
        scaled_shape = tuple([int(x * ratio) for x in old_size])

        interpolation = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
        img = cv2.resize(img, (scaled_shape[1], scaled_shape[0]), interpolation=interpolation)

        delta_w = new_size - scaled_shape[1]
        delta_h = new_size - scaled_shape[0]
        top,    bottom = delta_h // 2, delta_h - (delta_h // 2)
        left,   right  = delta_w // 2, delta_w - (delta_w // 2)

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)

        # CHW, float32, [0, 1] — NO ImageNet normalization
        blob = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return blob  # shape (3, H, W)

    def check(self, img, bbox, threshold=0.7):
        """
        img:    BGR numpy array (original frame)
        bbox:   face bounding box as (x1, y1, x2, y2) numpy array
        Returns: (is_real, score, message)
        """
        if self.session is None:
            return True, 1.0, "Model missing"

        try:
            # img comes in as BGR from OpenCV — convert to RGB for the model
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            face_crop = self._crop(img_rgb, bbox)
            if face_crop.size == 0:
                return False, 0.0, "Crop failed"

            blob = self._preprocess(face_crop)          # (3, 128, 128)
            blob = np.expand_dims(blob, 0)              # (1, 3, 128, 128)

            outputs = self.session.run(None, {self.input_name: blob})
            pred = outputs[0][0]  # shape (2,)

            # index 0 = real logit, index 1 = spoof logit (matches demo)
            real_logit  = float(pred[0])
            spoof_logit = float(pred[1])
            logit_diff  = real_logit - spoof_logit

            # Convert probability threshold -> logit space (same as demo)
            p = max(1e-6, min(1 - 1e-6, threshold))
            logit_threshold = float(np.log(p / (1 - p)))

            is_real = logit_diff >= logit_threshold
            score   = float(1.0 / (1.0 + np.exp(-logit_diff)))  # sigmoid

            print(f"[FAS] real={real_logit:.3f} spoof={spoof_logit:.3f} "
                  f"diff={logit_diff:.3f} thresh={logit_threshold:.3f} "
                  f"-> {'REAL' if is_real else 'SPOOF'} (score={score:.3f})")

            return is_real, score, "Real" if is_real else f"Spoof ({score:.2f})"

        except Exception as e:
            print(f"[FAS] Error during check: {e}")
            return True, 0.0, "FAS Error"