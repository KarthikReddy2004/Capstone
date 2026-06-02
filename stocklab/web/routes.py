"""HTTP API routes.

Endpoints are intentionally thin: they validate input, talk to the data loaders
and the job manager, and return JSON. All heavy lifting happens in worker threads
managed by the job manager, with state in Redis.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request, send_from_directory

from ..data.loaders import load_from_yahoo, load_from_csv
from ..data.features import prepare_features, build_summary
from ..jobs import get_job_manager
from ..pipeline import run_benchmark, MODEL_NAMES, MODEL_GROUPS
from ..storage import get_storage

api = Blueprint("api", __name__)


@api.get("/api/health")
def health():
    storage = get_storage()
    return jsonify({"ok": True, "redis": storage.ping(),
                    "models": len(MODEL_NAMES), "groups": MODEL_GROUPS})


@api.get("/api/models")
def models():
    return jsonify({"models": MODEL_NAMES, "groups": MODEL_GROUPS})


@api.post("/api/fetch_data")
def fetch_data():
    body = request.get_json(silent=True) or {}
    symbol = str(body.get("symbol", "AAPL")).upper().strip()
    start = str(body.get("start", "2018-01-01"))
    try:
        df = load_from_yahoo(symbol, start)
        prepared = prepare_features(df)
        key = get_job_manager().cache_dataframe(df, symbol)
        return jsonify({"success": True, "symbol": symbol,
                        "summary": build_summary(prepared), "cache_key": key})
    except Exception as err:  # noqa: BLE001
        return jsonify({"error": str(err)}), 400


@api.post("/api/upload_data")
def upload_data():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["file"]
    try:
        df = load_from_csv(file)
        prepared = prepare_features(df)
        key = get_job_manager().cache_dataframe(df, "upload")
        return jsonify({"success": True, "symbol": file.filename,
                        "summary": build_summary(prepared), "cache_key": key})
    except Exception as err:  # noqa: BLE001
        return jsonify({"error": str(err)}), 400


@api.post("/api/run")
def run():
    body = request.get_json(silent=True) or {}
    cache_key = body.get("cache_key")
    cfg = body.get("cfg", {})
    jm = get_job_manager()
    if not cache_key or not jm.has_dataframe(cache_key):
        return jsonify({"error": "Dataset not found. Fetch or upload first."}), 400
    df = jm.load_dataframe(cache_key)
    job_id = jm.submit(run_benchmark, len(MODEL_NAMES), df, cfg)
    return jsonify({"job_id": job_id})


@api.get("/api/status/<job_id>")
def status(job_id):
    payload = get_job_manager().get_status(job_id)
    if payload is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(payload)


@api.get("/api/results/<job_id>")
def results(job_id):
    payload = get_job_manager().get_results(job_id)
    if payload is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(payload)


def register_routes(app):
    app.register_blueprint(api)

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/<path:path>")
    def static_proxy(path):
        return send_from_directory(app.static_folder, path)
