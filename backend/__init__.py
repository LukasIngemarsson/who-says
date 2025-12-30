"""
WhoSays Backend - Flask Application Factory
"""
import os
import warnings

from flask import Flask
from dotenv import load_dotenv
from loguru import logger

# Configure file logging
os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv(".env")


def create_app():
    """Create and configure the Flask application."""
    static_folder_path = os.environ.get("FLASK_STATIC_FOLDER", "../frontend/dist")
    static_folder_path = os.path.abspath(static_folder_path)

    app = Flask(__name__, static_folder=static_folder_path, static_url_path='')

    # Import and initialize config/pipeline
    import backend.config as cfg
    cfg.init_pipeline()

    # Load tuning presets
    from backend.tuning import load_tuning_presets
    load_tuning_presets()

    # Load speaker embeddings
    if cfg.PIPELINE_AVAILABLE:
        from backend.speaker import load_embeddings
        load_embeddings()

    # Register blueprints
    from backend.routes import (
        static_bp,
        status_bp,
        tuning_bp,
        speakers_bp,
        overlap_bp,
        session_bp,
        process_bp,
    )

    app.register_blueprint(static_bp)
    app.register_blueprint(status_bp)
    app.register_blueprint(tuning_bp)
    app.register_blueprint(speakers_bp)
    app.register_blueprint(overlap_bp)
    app.register_blueprint(session_bp)
    app.register_blueprint(process_bp)

    return app
