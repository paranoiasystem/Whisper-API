from flask import Flask, abort, request
from tempfile import NamedTemporaryFile
import whisper
import torch
import time
import os

# Check if the WHISPER_MODEL environment variable is set
if "WHISPER_MODEL" not in os.environ:
    print("WHISPER_MODEL environment variable is not set")
    exit(1)

# Check if NVIDIA GPU is available
torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
match os.environ["WHISPER_MODEL"]:
    case "tiny.en" | "tiny" | "base.en" | "base" | "small.en" | "small" | "medium.en" | "medium" | "large":
        model = whisper.load_model(os.environ["WHISPER_MODEL"], device=DEVICE)
    case _:
        print("Invalid model name")
        exit(1)

app = Flask(__name__)

@app.route("/")
def root():
    return "Whisper is Running on {}".format(DEVICE)


@app.route('/whisper', methods=['POST'])
def whisper_api():
    startTime = time.time()

    if not request.files:
        abort(400)

    results = []

    for filename, handle in request.files.items():
        temp = NamedTemporaryFile()
        handle.save(temp)

        result = model.transcribe(temp.name)
        results.append({
            'filename': filename,
            'transcript': result['text'],
        })

    endTime = time.time()

    return {'results': results, 'time': endTime - startTime}