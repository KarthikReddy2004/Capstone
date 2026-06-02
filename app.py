"""Application entry point for the Adaptive Dual-Phase PSO-GWO Forecasting Lab.

Run locally:
    uv run python app.py            # then open http://127.0.0.1:5000

Production (gunicorn / waitress) should import ``create_app`` from
``stocklab.web`` and serve the returned WSGI app. Redis must be reachable via the
``STOCKLAB_REDIS_URL`` environment variable.
"""

from stocklab.web import create_app

app = create_app()

if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", "5000"))
    # threaded=True lets status polling stay responsive while a benchmark runs.
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
