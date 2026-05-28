from flask import Flask, Response, Response, render_template, request, jsonify, stream_with_context
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)
PROMPT = ChatPromptTemplate.from_messages([
    ("human", "다음 텍스트를 핵심 bullet 5개로 한국어 요약해줘.\n\n텍스트:\n{content}")
])
llm = ChatOllama(
    model = "gemma3:4b",
    temperature = 0.2,
    # base_url = "http://host.docker.internal:11434"
    base_url = "http://192.168.24.184:11434")
chain = PROMPT | llm | StrOutputParser()

def _sse_format(data: str, event: str | None = None) -> str:
    lines = []
    if event:
        lines.append(f"\nevent: {event}\n")
    else:
        lines.append(f"{data}")
    return "".join(lines)

@app.route("/summarize/stream", methods=["POST"])
def summarize_stream():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text 필드에 요약할 문자열을 넣어주세요."}), 400
    @stream_with_context
    def generate():
        yield _sse_format("started", event="start")
        try:
            for chunk in chain.stream({"content": text}):
                yield _sse_format(chunk)
        except Exception as e:
            yield _sse_format(f"error: {str(e)}", event="error")
        else:
            yield _sse_format("end", event="end")

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
    }
    return Response(generate(), mimetype="text/event-stream", headers=headers)

@app.route("/", methods=["GET"])
def index():
    return render_template("appsse.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)