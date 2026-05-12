import os


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    # Relative to project root (resolved in create_app); override with absolute paths if needed.
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    GENERATED_FOLDER = os.getenv("GENERATED_FOLDER", "static/generated")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
