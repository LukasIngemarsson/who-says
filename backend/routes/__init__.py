"""
Flask route blueprints for the WhoSays backend.
"""
from backend.routes.static import static_bp
from backend.routes.status import status_bp
from backend.routes.tuning import tuning_bp
from backend.routes.speakers import speakers_bp
from backend.routes.overlap import overlap_bp
from backend.routes.session import session_bp

__all__ = [
    'static_bp',
    'status_bp',
    'tuning_bp',
    'speakers_bp',
    'overlap_bp',
    'session_bp',
]
