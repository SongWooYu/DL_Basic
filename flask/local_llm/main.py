from flask import Flask, Response, request, jsonify, stream_with_context, render_template
import requests
app = Flask(__name__)

OLLAMA_URL = "http://192.168.24.184:11434/api/generate"

app = Flask(__name__)

@app.get('/')
def index():
    return render_template('index2.html')

@app.get("/2")
def index2():
    return render_template("index3.html")

@app.post('/api/generate')
def generate_stream():
    body = request.get_json(force=True, silent=True) or {}
    model = body.get('model', 'gemma4:e2b')
    prompt = body.get('prompt', "")
    stream = True
    upstream = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": stream},
        stream=True,
        timeout=600,
    )
    def gen():
        for line in upstream.iter_lines():
            if not line:
                continue
            yield line + b"\n"
    return Response(stream_with_context(gen()), mimetype="application/x-ndjson")

OLLAMA_CHAT_URL = "http://192.168.24.184:11434/api/chat"
@app.post("/api/chat")
def chat_stream():
    body = request.get_json(force=True, silent=True) or {}
    model = body.get('model', 'gemma4:e2b')
    messages = body.get('messages', [])
    options = body.get('options')
    upstream = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": model,
            "messages": messages,
            "stream" : True,
            **({"options": options} if options else {}),
        },
        stream=True,
        timeout=600,
    )
    upstream.raise_for_status()

    def generate():
        for line in upstream.iter_lines():
            if not line:
                continue
            yield line + b"\n"
    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)