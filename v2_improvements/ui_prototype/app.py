"""Standalone localhost server exposing computed V2 pipeline output."""
from pathlib import Path
from flask import Flask, jsonify, send_from_directory
import json
ROOT=Path(__file__).resolve().parent; app=Flask(__name__)
@app.get("/api/dashboard")
def dashboard(): return jsonify(json.loads((ROOT/"dashboard_data.json").read_text(encoding="utf-8")))
@app.get("/")
def index(): return send_from_directory(ROOT,"index.html")
@app.get("/<path:path>")
def files(path): return send_from_directory(ROOT,path)
if __name__=="__main__": app.run(debug=True,port=5052)
