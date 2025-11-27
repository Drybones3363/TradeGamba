

MODEL_FILE = "BruteforceC"


from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from learningAI import SetupEnv, StrategyConfig, ScoringModel
from AITranscribe import *
from flask_cors import CORS
import os

def extractFeatures(bars):
    return extract_features_raw_ohlcv(bars)

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app, resources={r"/*": {"origins": "*"}})
app.config.update(
    SEND_FILE_MAX_AGE_DEFAULT=0,
    TEMPLATES_AUTO_RELOAD=True
)
app.jinja_env.cache = {}

cfg = StrategyConfig()
learner = SetupEnv(cfg)
model = ScoringModel()  # add this near learner initialization
model.agent.load("Models/"+MODEL_FILE+".pkl")

@app.route("/score", methods=["POST"])
def score():
    data = request.get_json()
    bars = data["bars"]
    feats = extractFeatures(bars)
    score_val = model.score_setup(feats)
    print(score_val)
    return jsonify({"score": score_val})

@app.route("/train", methods=["POST"])
def train():
    data = request.get_json()
    bars = data["bars"]
    rewardInfo = data["rewards"]
    entry = float(data["entry"])

    # future highs/lows after the entry bar
    highs = np.array([b['high'] for b in bars[1:]], dtype=np.float32)
    lows  = np.array([b['low']  for b in bars[1:]], dtype=np.float32)

    # features from the full window you sent
    feats = extractFeatures(bars)

    # compute rewards for each possible action at this setup
    actions = [-1, 0, 1]  # short / skip / long
    rewards = [
        learner.step(-1, rewardInfo),
        learner.step( 0, rewardInfo),
        learner.step(+1, rewardInfo),
    ]

    # train on this one setup by duplicating features for each action
    model.train_on_batch([feats, feats, feats], actions, rewards, iters=1)
    #model.agent.save("Models/"+MODEL_FILE+".pkl")

    # optional: return updated score for UI
    new_score = model.score_setup(feats)
    return jsonify({"rewards": rewards, "score": new_score})

@app.after_request
def no_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

def deploy():
    app.run(host="0.0.0.0", port=5000)  # dev only


if __name__ == "__main__":
    deploy()
