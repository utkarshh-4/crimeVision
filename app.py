from __future__ import annotations

"""CrimeVision AI application entrypoint and Flask factory."""

import logging
import os
from flask import Flask, jsonify, render_template
from werkzeug.exceptions import HTTPException

import config
from modules.utils import ensure_dirs
from routes.api import api_bp


def create_app() -> Flask:
    """Create and configure the CrimeVision AI Flask application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Ensure runtime data, upload, session, and report directories exist
    ensure_dirs()

    app = Flask(__name__, template_folder="templates", static_folder="static")

    # Configure maximum upload size
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024

    # Register API blueprint
    app.register_blueprint(api_bp)

    @app.route("/", methods=["GET"])
    def index():
        """Serve the CrimeVision AI dashboard template."""
        return render_template("index.html")

    # JSON error handlers
    @app.errorhandler(400)
    def bad_request(error):
        msg = error.description if isinstance(error, HTTPException) and error.description else "Bad Request"
        return jsonify({"error": msg}), 400

    @app.errorhandler(404)
    def not_found(error):
        msg = error.description if isinstance(error, HTTPException) and error.description else "Not Found"
        return jsonify({"error": msg}), 404

    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({"error": f"File size exceeds maximum upload limit of {config.MAX_UPLOAD_MB}MB"}), 413

    @app.errorhandler(500)
    def internal_server_error(error):
        return jsonify({"error": "Internal Server Error"}), 500

    return app


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app = create_app()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
