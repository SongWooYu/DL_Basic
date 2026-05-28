from flask import Flask, Response, Response, render_template, request, jsonify, stream_with_context
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

app = Flask(__name__)

_json_llm = ChatOllama(
    model = "gemma3:4b",
    temperature = 0.2,
    # base_url = "http://host.docker.internal:11434"
    base_url = "http://192.168.24.184:11434")
_json_parser = JsonOutputParser()
_json_format = _json_parser.get_format_instructions()

_json_prompt = ChatPromptTemplate.from_messages([
    ("human", """다음 텍스트를 JSON으로 요약해
- "summary" : 한국어 핵심 요약(3~5문장)
- "keywords" : 핵심 키워드 5~8개 리스트
- "oneline" : 한 줄 요약(20자 내외)
반드시 JSON만 출력.
{format_instructions}

텍스트:
{content}
""")
])
_json_chain = _json_prompt | _json_llm | _json_parser

@app.route("/summarize/json", methods=["POST"])
def summarize_structured():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text 필드가 필요합니다."}), 400
    result = _json_chain.invoke({"content": text, "format_instructions": _json_format})
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)