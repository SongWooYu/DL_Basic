# 식인종-선교사 LLM 가이드 - 턴 단위 자동 실행 패치

## 변경점

- 기존 `/api/auto-run`은 서버 요청 하나가 끝날 때까지 브라우저가 기다려서 멈춘 것처럼 보일 수 있었습니다.
- 이번 버전은 `/api/auto-step`을 추가했습니다.
- 브라우저가 자동 실행을 1턴씩 요청하므로, 매 턴의 AI 판단/Command/검증 결과가 화면에 바로 누적됩니다.

## 실행

```bash
pip install -r requirements.txt
python main.py
```

접속:

```text
http://127.0.0.1:5000/3
```

## 설정

`main.py` 상단의 Ollama 주소와 모델명을 환경에 맞게 확인하십시오.

```python
OLLAMA_CHAT_URL = "http://192.168.24.184:11434/api/chat"
MODEL = "gemma4:e2b"
```
