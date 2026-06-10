from __future__ import annotations

import os
import secrets
from typing import Dict, List

from flask import Flask, Response, jsonify, render_template, request, session, stream_with_context
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")
app.config["SESSION_COOKIE_NAME"] = "flask_langchain_chat_sid"
app.config["PERMANENT_SESSION_LIFETIME"] = 60 * 60 * 24 * 7

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://192.168.24.184:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")

SYSTEM_PROMPT = """
너는 Flask 세션별 메모리를 적용한 학습용 ChatBot이다.
답변은 한국어로 한다.
사용자의 이전 대화를 참고하되, 모르면 모른다고 답한다.
답변은 과제 제출용 실습 수준에서 간결하게 작성한다.
""".strip()

# 서버 프로세스가 살아 있는 동안 유지되는 세션별 대화 저장소입니다.
# 브라우저를 닫았다 열어도 쿠키에 sid가 남아 있으면 같은 대화를 복구합니다.
# 단, Flask 서버를 재시작하면 이 메모리 저장소는 초기화됩니다.
CHAT_STORE: Dict[str, List[dict]] = {}


def get_sid() -> str:
    session.permanent = True
    sid = session.get("sid")
    if not sid:
        sid = secrets.token_urlsafe(16)
        session["sid"] = sid
    CHAT_STORE.setdefault(sid, [])
    return sid


def to_langchain_messages(history: List[dict], new_user_text: str):
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for item in history:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            messages.append(AIMessage(content=item["content"]))
    messages.append(HumanMessage(content=new_user_text))
    return messages


def create_llm() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
    )


@app.get("/")
def index():
    get_sid()
    return render_template("index.html", model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)


@app.get("/api/history")
def api_history():
    sid = get_sid()
    return jsonify({"sid": sid, "history": CHAT_STORE.get(sid, [])})


@app.post("/api/reset")
def api_reset():
    sid = get_sid()
    CHAT_STORE[sid] = []
    return jsonify({"ok": True, "message": "이전 대화를 삭제했습니다.", "history": []})


@app.post("/api/chat")
def api_chat():
    sid = get_sid()
    body = request.get_json(force=True, silent=True) or {}
    user_text = str(body.get("message", "")).strip()

    if not user_text:
        return jsonify({"ok": False, "message": "message가 비어 있습니다."}), 400

    history = CHAT_STORE.setdefault(sid, [])
    messages = to_langchain_messages(history, user_text)

    def event_stream():
        assistant_text = ""
        try:
            llm = create_llm()
            yield "event: meta\ndata: {\"status\": \"start\"}\n\n"

            for chunk in llm.stream(messages):
                token = getattr(chunk, "content", "") or ""
                if not token:
                    continue
                assistant_text += token
                safe = token.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "")
                safe = safe.replace('"', '\\"')
                yield f'event: token\ndata: {{"token": "{safe}"}}\n\n'

            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": assistant_text})
            yield "event: done\ndata: {\"status\": \"done\"}\n\n"
        except Exception as exc:
            message = str(exc).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
            yield f'event: error\ndata: {{"message": "{message}"}}\n\n'

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
