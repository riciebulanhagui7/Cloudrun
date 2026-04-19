from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/api/echo", methods=["GET"])
def echo():
    name = request.args.get("name", "Guest")
    return jsonify({
        "message": f"Hello {name}"
    })