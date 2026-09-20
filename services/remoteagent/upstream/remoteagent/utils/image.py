from __future__ import annotations

import base64
import mimetypes
import os


class ImageUtils:
    @staticmethod
    def encode_image(path: str) -> str:
        path = path.strip().strip("'").strip('"')
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image not found: {path}")
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def image_data_url(path: str) -> str:
        """Return an OpenAI-compatible image data URL for a local image."""
        clean_path = path.strip().strip("'").strip('"')
        mime_type, _ = mimetypes.guess_type(clean_path)
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/png"
        return f"data:{mime_type};base64,{ImageUtils.encode_image(clean_path)}"


encode_image = ImageUtils.encode_image
image_data_url = ImageUtils.image_data_url
