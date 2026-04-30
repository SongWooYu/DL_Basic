from flask import Flask, Response, request, jsonify, stream_with_context, render_template
import requests
import json
import time
import os
import random
import re

app = Flask(__name__)

OLLAMA_URL = "http://192.168.24.184:11434/api/generate"
OLLAMA_CHAT_URL = "http://192.168.24.184:11434/api/chat"

COMFY_URL = "http://192.168.24.184:8188"
WORKFLOW_PATH = "./static/test.json"


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
    upstream.raise_for_status()

    def gen():
        for line in upstream.iter_lines():
            if not line:
                continue
            yield line + b"\n"

    return Response(stream_with_context(gen()), mimetype="application/x-ndjson")


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
            "stream": True,
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


# -----------------------------
# ComfyUI 관련 함수
# -----------------------------

def load_workflow(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Workflow not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_workflow(graph: dict, pos: str, neg: str, seed: int) -> dict:
    graph["6"]["inputs"]["text"] = pos
    graph["7"]["inputs"]["text"] = neg
    graph["3"]["inputs"]["seed"] = seed
    return graph


def submit_prompt(graph: dict) -> str:
    r = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": graph},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["prompt_id"]


def poll_history(prompt_id: str, timeout_sec: int = 180, interval: float = 1.0) -> dict | None:
    end = time.time() + timeout_sec

    while time.time() < end:
        r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30)

        if r.status_code == 200 and r.json():
            return list(r.json().values())[0]

        time.sleep(interval)

    return None


def extract_first_image(history_block: dict) -> str:
    if not history_block:
        return ""

    outputs = history_block.get("outputs", {})

    for node_id, node_out in outputs.items():
        if "images" in node_out and node_out["images"]:
            img = node_out["images"][0]
            fn = img.get("filename")
            sub = img.get("subfolder", "")
            img_type = img.get("type", "output")

            if fn:
                return f"{COMFY_URL}/view?filename={fn}&subfolder={sub}&type={img_type}"

    return ""


# -----------------------------
# LLM: 한글 요구사항 -> 이미지 프롬프트 변환
# -----------------------------

def make_image_prompt_with_llm(korean_request: str, model: str = "gemma4:e2b") -> tuple[str, str, str]:
    instruction = f"""
You are an expert Stable Diffusion prompt engineer.

Convert the following Korean image request into an English image generation prompt.

Return ONLY valid JSON.
Do not include markdown.
Do not include explanations.

JSON format:
{{
  "positive": "English positive prompt for Stable Diffusion",
  "negative": "English negative prompt"
}}

Rules:
- positive must be detailed, visual, and suitable for Stable Diffusion.
- include subject, background, style, lighting, composition, quality keywords.
- negative must include common bad image terms.
- Do not translate as a plain sentence. Make it an image generation prompt.

Korean request:
{korean_request}
"""

    r = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": instruction,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        },
        timeout=180,
    )
    r.raise_for_status()

    raw = r.json().get("response", "").strip()

    # 모델이 JSON 앞뒤로 쓸데없는 문장을 붙일 수 있어서 JSON 부분만 추출
    match = re.search(r"\{.*\}", raw, re.DOTALL)

    if match:
        try:
            data = json.loads(match.group(0))
            positive = data.get("positive", "").strip()
            negative = data.get("negative", "").strip()

            if positive:
                if not negative:
                    negative = "low quality, worst quality, blurry, text, watermark, logo, distorted, deformed"
                return positive, negative, raw
        except json.JSONDecodeError:
            pass

    # JSON 파싱 실패 시 fallback
    fallback_positive = raw if raw else korean_request
    fallback_negative = "low quality, worst quality, blurry, text, watermark, logo, distorted, deformed"
    return fallback_positive, fallback_negative, raw


def generate_image_with_comfy(pos: str, neg: str) -> str:
    seed = random.randint(0, 112589906842624)

    graph = load_workflow(WORKFLOW_PATH)
    graph = update_workflow(graph, pos, neg, seed)

    prompt_id = submit_prompt(graph)
    history_block = poll_history(prompt_id)

    return extract_first_image(history_block)


# -----------------------------
# 과제용 페이지
# -----------------------------

@app.route("/3", methods=["GET", "POST"])
def llm_comfy_image():
    korean_request = ""
    model = "gemma4:e2b"
    pos = ""
    neg = "low quality, worst quality, blurry, text, watermark"
    raw_llm = ""
    img_url = ""
    error = ""

    if request.method == "POST":
        korean_request = request.form.get("korean_request", "").strip()
        model = request.form.get("model", "gemma4:e2b").strip() or "gemma4:e2b"

        if not korean_request:
            error = "한글 요구사항을 입력하세요."
        else:
            try:
                pos, neg, raw_llm = make_image_prompt_with_llm(korean_request, model)
                img_url = generate_image_with_comfy(pos, neg)

                if not img_url:
                    error = "이미지 생성은 요청됐지만 결과 이미지 URL을 찾지 못했습니다."

            except Exception as e:
                error = str(e)

    return render_template(
        "llm_image.html",
        korean_request=korean_request,
        model=model,
        pos=pos,
        neg=neg,
        raw_llm=raw_llm,
        img_url=img_url,
        error=error,
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)