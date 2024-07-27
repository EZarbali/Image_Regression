import io
import json
import os
import traceback

from PIL import Image
from flask import Flask, jsonify, request
import sys

sys.path.append("/app")
from model_inferrer import Model_Inferrer

app = Flask(__name__)
APP_ROOT = os.getenv("APP_ROOT", "/infer")
HOST = "0.0.0.0"
PORT_NUMBER = int(os.getenv("PORT_NUMBER", 8080))

model_inferrer = Model_Inferrer()


@app.route("/", methods=["GET"])
def root():
    return jsonify({"msg": "POST an image of a car to the /infer endpoint"})


@app.route("/infer", methods=["POST"])
def infer():
    if request.method == "POST":
        file = request.files["file"]
        if file is not None:
            pred1, pred2 = model_inferrer.inference(file)
            return jsonify({"pred_hood": pred1, "pred_backdoor_left": pred2})


@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify(stackTrace=traceback.format_exc())


if __name__ == "__main__":
    app.run(host=HOST, port=8080)
