import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from rag import load_vector_store
from nova import process_query

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# Initialize RAG vector store on startup
print("[NOVA] Initializing vector store...")
db = load_vector_store()
print("[NOVA] NOVA is ready to answer questions about the cosmos!")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data["message"].strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    result = process_query(user_message, db)
    return jsonify({
        "answer": result["answer"],
        "scenario": result["scenario"],
        "label": result["label"]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "NOVA online", "specialty": "Space exploration & astronomy"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
