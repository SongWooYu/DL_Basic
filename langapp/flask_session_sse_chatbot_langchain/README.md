# Flask Session Memory ChatBot with LangChain

## 과제 요구사항

1. Cookie를 사용하여 브라우저가 닫혀도 세션 복구
2. 사용자 요청에 의해 이전 대화 삭제
3. 스트리밍(SSE) 구현으로 채팅 내용을 실시간 표현
4. LangChain을 사용하여 Ollama LLM 호출

## 구조

```text
Browser
  - Cookie: sid 보관
  - SSE 응답 실시간 표시
    ↓
Flask Server
  - Flask session으로 sid 발급/복구
  - CHAT_STORE[sid]에 대화 기록 저장
  - LangChain ChatOllama로 메시지 전달
    ↓
Ollama
  - gemma4:e2b 모델 응답 생성
```

## 실행

```bash
pip install -r requirements.txt
python app.py
```

브라우저 접속:

```text
http://127.0.0.1:5000
```

## 환경변수

```bash
set OLLAMA_BASE_URL=http://192.168.24.184:11434
set OLLAMA_MODEL=gemma4:e2b
python app.py
```

Linux/macOS:

```bash
export OLLAMA_BASE_URL=http://192.168.24.184:11434
export OLLAMA_MODEL=gemma4:e2b
python app.py
```

## 한계

현재 대화 저장소는 Flask 서버 프로세스 메모리입니다.
따라서 브라우저를 닫았다 열어도 쿠키 sid로 복구되지만, Flask 서버를 재시작하면 대화 기록은 사라집니다.
DB나 Redis는 수업 범위를 벗어날 수 있으므로 사용하지 않았습니다.
