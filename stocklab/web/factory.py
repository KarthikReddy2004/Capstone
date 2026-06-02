"""Flask application factory."""

from __future__ import annotations

import os
import warnings

from flask import Flask

from .routes import register_routes


def create_app() -> Flask:
    warnings.filterwarnings("ignore")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    static_dir = os.path.join(base_dir, "static")
    app = Flask(__name__, static_folder=static_dir, static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB CSV cap
    app.config["JSON_SORT_KEYS"] = False
    register_routes(app)
    # Eagerly construct the job manager (and Redis singleton) so connection
    # failures surface at boot rather than on first request.
    from ..jobs import get_job_manager
    get_job_manager()
    return app
