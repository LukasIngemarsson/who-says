"""
WhoSays Application Entry Point

This file serves as the main entry point for the WhoSays backend.
All application logic has been moved to the 'backend' package.
"""
from backend import create_app

app = create_app()

if __name__ == "__main__":
    # threaded=False prevents concurrent request issues with PyTorch models
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=False)
