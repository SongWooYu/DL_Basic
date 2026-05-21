from flask import Flask, request, jsonify, render_template
from pathlib import Path
import re
import requests

app = Flask(__name__)

OLLAMA_CHAT_URL = "http://192.168.24.184:11434/api/chat"
MODEL = "gemma4:e2b"

BASE_DIR = Path(__file__).resolve().parent
GAME_GUIDE_PATH = BASE_DIR / "game_guide.md"

TOTAL_M = 3
TOTAL_C = 3

COMMANDS = {
    "1M": (1, 0),
    "2M": (2, 0),
    "1C": (0, 1),
    "2C": (0, 2),
    "1M1C": (1, 1),
}
COMMAND_ORDER = ["1M", "2M", "1C", "2C", "1M1C"]

GAME_KEYWORDS = [
    "식인종", "선교사", "게임", "배", "강", "왼쪽", "오른쪽", "상태", "현재", "상황",
    "이동", "커맨드", "command", "명령", "1m", "2m", "1c", "2c", "1m1c",
    "다음", "이번", "어떻게", "안전", "승리", "게임오버", "뭐", "해야", "추천",
    "턴", "수", "요약", "정리", "앞으로", "시나리오", "선택지", "탈락", "자동"
]
FOLLOWUP_WORDS = ["다음", "이번", "그다음", "계속", "이번에는", "다음은", "이제", "또", "진행"]
INJECTION_KEYWORDS = [
    "규칙 무시", "지시 무시", "시스템 프롬프트", "system prompt",
    "너는 이제", "역할 바꿔", "프롬프트 보여", "제약 무시"
]

status = {"left_m": 0, "left_c": 0, "right_m": 3, "right_c": 3, "boat": "right"}
history = []
game_finished = False
game_finished_message = ""

# Flask 서버가 살아 있는 동안 보존되는 기록입니다.
move_log = []        # 실제 실행된 Command
failed_log = []      # LLM/자동 실행이 제안했지만 탈락한 Command
auto_trace = []      # 자동 실행 과정 전체
visited_runtime_states = {(0, 0, 3, 3, "right")}


def load_game_guide():
    try:
        return GAME_GUIDE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "식인종-선교사 게임 규칙 파일을 찾지 못했습니다. 기본 규칙을 사용합니다."


def normalize_command(command):
    return str(command).strip().upper().replace(" ", "")


def extract_command(text):
    if not text:
        return None
    matches = re.findall(r"\b(1M1C|1M|2M|1C|2C)\b", text.upper())
    return matches[-1] if matches else None


def is_safe(missionary, cannibal):
    return missionary == 0 or missionary >= cannibal


def is_valid_people_count(s):
    values = [s["left_m"], s["left_c"], s["right_m"], s["right_c"]]
    if any(v < 0 for v in values):
        return False
    if s["left_m"] + s["right_m"] != TOTAL_M:
        return False
    if s["left_c"] + s["right_c"] != TOTAL_C:
        return False
    return True


def judge(s):
    if not is_valid_people_count(s):
        return True, "게임오버: 전체 인원 수가 맞지 않습니다."
    if not is_safe(s["left_m"], s["left_c"]):
        return True, "게임오버: 왼쪽에서 식인종 수가 선교사 수보다 많습니다."
    if not is_safe(s["right_m"], s["right_c"]):
        return True, "게임오버: 오른쪽에서 식인종 수가 선교사 수보다 많습니다."
    if s["left_m"] == TOTAL_M and s["left_c"] == TOTAL_C:
        return True, "승리: 모두 왼쪽으로 이동했습니다."
    return False, "계속 진행"


def game(s, command):
    command = normalize_command(command)
    if command not in COMMANDS:
        return s, False, "입력 오류: 사용할 수 없는 Command입니다."

    move_m, move_c = COMMANDS[command]
    next_status = s.copy()

    if s["boat"] == "right":
        if s["right_m"] < move_m or s["right_c"] < move_c:
            return s, False, "이동 불가: 오른쪽에 해당 사람 수가 부족합니다."
        next_status["right_m"] -= move_m
        next_status["right_c"] -= move_c
        next_status["left_m"] += move_m
        next_status["left_c"] += move_c
        next_status["boat"] = "left"
    else:
        if s["left_m"] < move_m or s["left_c"] < move_c:
            return s, False, "이동 불가: 왼쪽에 해당 사람 수가 부족합니다."
        next_status["left_m"] -= move_m
        next_status["left_c"] -= move_c
        next_status["right_m"] += move_m
        next_status["right_c"] += move_c
        next_status["boat"] = "right"

    end, message = judge(next_status)
    return next_status, end, message


def status_to_tuple(s):
    return (s["left_m"], s["left_c"], s["right_m"], s["right_c"], s["boat"])


def status_to_key(s):
    return "|".join(map(str, status_to_tuple(s)))


def status_text(s):
    return (
        f"왼쪽: 선교사 {s['left_m']}명, 식인종 {s['left_c']}명 / "
        f"오른쪽: 선교사 {s['right_m']}명, 식인종 {s['right_c']}명 / "
        f"배 위치: {'왼쪽' if s['boat'] == 'left' else '오른쪽'}"
    )


def is_goal(s):
    return s["left_m"] == TOTAL_M and s["left_c"] == TOTAL_C


def is_game_question(text):
    t = text.lower()
    if any(k in t for k in GAME_KEYWORDS):
        return True
    if history or move_log:
        return any(w in t for w in FOLLOWUP_WORDS)
    return False


def is_injection(text):
    t = text.lower()
    return any(k in t for k in INJECTION_KEYWORDS)


def format_move_log(limit=12):
    if not move_log:
        return "- 아직 실행한 Command가 없습니다."
    rows = []
    for item in move_log[-limit:]:
        rows.append(
            f"- {item['turn']}턴: {item['command']} ({item['source']}) | "
            f"전: {status_text(item['before'])} -> 후: {status_text(item['after'])} | {item['message']}"
        )
    return "\n".join(rows)


def format_failed_log(limit=12):
    if not failed_log:
        return "- 아직 탈락한 선택지가 없습니다."
    rows = []
    for item in failed_log[-limit:]:
        rows.append(
            f"- {item['turn']}회차 후보 {item['command']} 탈락 | "
            f"상태: {status_text(item['state'])} | 이유: {item['reason']}"
        )
    return "\n".join(rows)


def append_move_log(source, command, before, after, message, end):
    entry = {
        "turn": len(move_log) + 1,
        "source": source,
        "command": command,
        "before": before.copy(),
        "after": after.copy(),
        "before_key": status_to_key(before),
        "after_key": status_to_key(after),
        "message": message,
        "end": end,
    }
    move_log.append(entry)
    visited_runtime_states.add(status_to_tuple(after))
    return entry


def append_failed_log(source, command, state, reason, raw_answer=""):
    entry = {
        "turn": len(failed_log) + 1,
        "source": source,
        "command": command,
        "state": state.copy(),
        "state_key": status_to_key(state),
        "reason": reason,
        "raw_answer": raw_answer,
    }
    failed_log.append(entry)
    auto_trace.append(entry)
    return entry


def call_ollama(messages, timeout=120):
    upstream = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=timeout,
    )
    upstream.raise_for_status()
    data = upstream.json()
    return data.get("message", {}).get("content", "")


def build_rule_context():
    return f"""
{load_game_guide()}

[현재 게임 상태]
{status_text(status)}

[실제 실행 로그]
{format_move_log()}

[실패/탈락 기록]
{format_failed_log()}
""".strip()


def build_guide_messages(user_text):
    system_prompt = f"""
너는 식인종-선교사 게임 전용 가이드 챗봇이다.

[역할]
- 사용자의 현재 게임 상태를 보고 다음에 시도할 Command를 하나 제안한다.
- 정답 경로 전체, 남은 전체 경로, 최단 턴 수는 말하지 않는다.
- 현재 턴에서 왜 그 Command가 좋아 보이는지만 짧게 설명한다.
- 게임과 관련 없는 질문은 거부한다.
- 실제 실행 여부와 성공/실패 판정은 서버가 한다. 너는 조언만 한다.

[참고 규칙과 현재 상태]
{build_rule_context()}

[답변 형식]
1. 현재 상태를 1~2문장으로 설명한다.
2. 이번 턴에 시도할 Command를 하나만 제안한다.
3. 마지막 줄은 반드시 다음 형식으로 끝낸다.
추천 Command: <1M|2M|1C|2C|1M1C>
""".strip()
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": user_text})
    return messages


def build_auto_messages(attempt_no):
    system_prompt = f"""
너는 식인종-선교사 게임을 한 턴씩 풀어가는 자동 플레이어다.
정답 경로를 미리 받지 않는다. 아래 규칙, 현재 상태, 실행 로그, 실패 로그만 보고 이번 턴의 Command 하나를 고른다.

[중요 규칙]
- 반드시 Command 하나만 고른다.
- 실패/탈락 기록에 있는 같은 상태의 같은 Command는 다시 고르지 않는다.
- 같은 상태로 되돌아가는 반복을 피한다.
- 설명은 짧게 한다.
- 마지막 줄은 반드시 다음 형식으로 끝낸다.
추천 Command: <1M|2M|1C|2C|1M1C>

[참고 규칙과 현재 상태]
{build_rule_context()}
""".strip()
    user_prompt = f"자동 실행 {attempt_no}회차다. 현재 상태에서 이번 턴에 실행할 Command 하나를 선택해라."
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def validate_candidate(command, before):
    command = normalize_command(command)
    if command not in COMMANDS:
        return False, before, False, "사용할 수 없는 Command입니다."

    after, end, message = game(before, command)

    if after == before and ("이동 불가" in message or "입력 오류" in message):
        return False, after, end, message
    if "게임오버" in message:
        return False, after, end, message
    if status_to_tuple(after) in visited_runtime_states and not is_goal(after):
        return False, after, end, "이미 지나간 상태로 돌아가므로 반복 가능성이 있습니다."

    return True, after, end, message


def auto_step_once(step_no=1, max_retry_per_state=None):
    """LLM에게 한 턴만 맡기고, 서버가 검증한다.
    긴 요청으로 브라우저가 멈춘 것처럼 보이는 문제를 막기 위해 /api/auto-step에서 사용한다.
    """
    global status, game_finished, game_finished_message

    if max_retry_per_state is None:
        max_retry_per_state = len(COMMAND_ORDER) + 2

    if game_finished:
        return {
            "ok": False,
            "done": True,
            "moved": False,
            "message": game_finished_message or "게임이 이미 종료되었습니다.",
            "answer": "",
            "command": None,
            "status": status,
        }

    if is_goal(status):
        game_finished = True
        game_finished_message = "승리: 모두 왼쪽으로 이동했습니다."
        return {
            "ok": True,
            "done": True,
            "moved": False,
            "message": game_finished_message,
            "answer": "",
            "command": None,
            "status": status,
        }

    state_key = status_to_key(status)

    for retry in range(1, max_retry_per_state + 1):
        try:
            answer = call_ollama(build_auto_messages(step_no), timeout=45)
        except Exception as e:
            append_failed_log("auto", "LLM", status, f"LLM 호출 실패: {e}")
            return {
                "ok": False,
                "done": True,
                "moved": False,
                "message": "자동 실행 중 LLM 호출에 실패했습니다.",
                "answer": str(e),
                "command": None,
                "status": status,
            }

        cmd = extract_command(answer)
        if not cmd:
            append_failed_log("auto", "없음", status, "LLM 응답에서 Command를 찾지 못했습니다.", answer)
            continue

        already_failed = any(
            item["state_key"] == state_key and item["command"] == cmd
            for item in failed_log
        )
        if already_failed:
            append_failed_log("auto", cmd, status, "같은 상태에서 이미 탈락한 Command를 다시 제안했습니다.", answer)
            continue

        before = status.copy()
        ok, after, end, message = validate_candidate(cmd, before)
        if not ok:
            append_failed_log("auto", cmd, before, message, answer)
            continue

        status = after
        append_move_log("auto", cmd, before, status, message, end)

        if end:
            game_finished = True
            game_finished_message = message

        return {
            "ok": True,
            "done": game_finished,
            "moved": True,
            "message": message,
            "answer": answer,
            "command": cmd,
            "status": status,
        }

    append_failed_log("auto", "중단", status, "현재 상태에서 LLM이 유효한 새 선택지를 찾지 못해 자동 실행을 중단했습니다.")
    return {
        "ok": False,
        "done": True,
        "moved": False,
        "message": "현재 상태에서 LLM이 유효한 새 선택지를 찾지 못해 자동 실행을 중단했습니다.",
        "answer": "",
        "command": None,
        "status": status,
    }


def build_game_report(short=False):
    lines = []
    lines.append("[현재까지 실행 기록]")
    if not move_log:
        lines.append("- 아직 실행한 Command가 없습니다.")
    else:
        for item in move_log:
            lines.append(
                f"- {item['turn']}턴: {item['command']} ({item['source']}) | "
                f"{status_text(item['before'])} -> {status_text(item['after'])} | {item['message']}"
            )

    lines.append("")
    lines.append("[탈락한 선택지 기록]")
    if not failed_log:
        lines.append("- 아직 탈락한 선택지가 없습니다.")
    else:
        for item in failed_log:
            lines.append(
                f"- {item['turn']}회차: {item['command']} ({item['source']}) | "
                f"{status_text(item['state'])} | {item['reason']}"
            )

    lines.append("")
    lines.append("[현재 상태]")
    lines.append(f"- {status_text(status)}")

    lines.append("")
    lines.append("[결과]")
    if game_finished:
        lines.append(f"- {game_finished_message}")
    else:
        lines.append("- 아직 게임이 종료되지 않았습니다. 다음 턴을 계속 진행해야 합니다.")

    return "\n".join(lines)


@app.get("/")
def index():
    return render_template("index3.html")


@app.get("/1")
def index1():
    return render_template("index.html")


@app.get("/2")
def index2():
    return render_template("index2.html")


@app.get("/3")
def index3():
    return render_template("index3.html")


@app.get("/api/status")
def api_status():
    return jsonify({
        **status,
        "game_finished": game_finished,
        "game_finished_message": game_finished_message,
        "move_log": move_log,
        "failed_log": failed_log,
        "auto_trace_count": len(auto_trace),
    })


@app.post("/api/reset")
def api_reset():
    global status, history, game_finished, game_finished_message, move_log, failed_log, auto_trace, visited_runtime_states
    status = {"left_m": 0, "left_c": 0, "right_m": 3, "right_c": 3, "boat": "right"}
    history = []
    move_log = []
    failed_log = []
    auto_trace = []
    visited_runtime_states = {status_to_tuple(status)}
    game_finished = False
    game_finished_message = ""
    return jsonify({
        "status": status,
        "message": "게임을 초기화했습니다.",
        "game_finished": game_finished,
        "game_finished_message": game_finished_message,
        "move_log": move_log,
        "failed_log": failed_log,
    })


@app.post("/api/move")
def api_move():
    global status, game_finished, game_finished_message

    if game_finished:
        return jsonify({
            "status": status,
            "end": True,
            "message": game_finished_message or "게임이 종료되었습니다. 초기화만 가능합니다.",
            "game_finished": True,
            "game_finished_message": game_finished_message,
            "move_log": move_log,
            "failed_log": failed_log,
        })

    body = request.get_json(force=True, silent=True) or {}
    command = normalize_command(body.get("command", ""))
    before = status.copy()
    after, end, message = game(status, command)

    # 수동 조작은 사용자가 실제로 누른 선택이므로 기존 게임 규칙대로 상태를 반영한다.
    status = after
    append_move_log("manual", command, before, status, message, end)

    if end:
        game_finished = True
        game_finished_message = message

    return jsonify({
        "status": status,
        "end": end,
        "message": message,
        "game_finished": game_finished,
        "game_finished_message": game_finished_message,
        "move_log": move_log,
        "failed_log": failed_log,
    })


@app.post("/api/auto-step")
def api_auto_step():
    body = request.get_json(force=True, silent=True) or {}
    step_no = int(body.get("step", len(move_log) + 1))

    result = auto_step_once(step_no=step_no)
    report = build_game_report(short=False)

    return jsonify({
        "ok": result["ok"],
        "done": result["done"],
        "moved": result["moved"],
        "message": result["message"],
        "answer": result["answer"],
        "command": result["command"],
        "status": status,
        "game_finished": game_finished,
        "game_finished_message": game_finished_message,
        "report": report,
        "move_log": move_log,
        "failed_log": failed_log,
        "auto_trace": auto_trace,
    })


@app.post("/api/auto-run")
def api_auto_run():
    global status, game_finished, game_finished_message

    if game_finished:
        return jsonify({
            "ok": False,
            "message": game_finished_message or "게임이 이미 종료되었습니다. 초기화 후 다시 실행하세요.",
            "status": status,
            "game_finished": game_finished,
            "game_finished_message": game_finished_message,
            "report": build_game_report(short=False),
            "move_log": move_log,
            "failed_log": failed_log,
            "auto_trace": auto_trace,
        })

    max_steps = 40
    max_retry_per_state = len(COMMAND_ORDER) + 2
    executed = []

    for step in range(1, max_steps + 1):
        if is_goal(status):
            game_finished = True
            game_finished_message = "승리: 모두 왼쪽으로 이동했습니다."
            break

        state_key = status_to_key(status)
        state_retry = 0
        moved = False

        while state_retry < max_retry_per_state:
            state_retry += 1
            try:
                answer = call_ollama(build_auto_messages(step), timeout=45)
            except Exception as e:
                append_failed_log("auto", "LLM", status, f"LLM 호출 실패: {e}")
                report = build_game_report(short=False)
                history.append({"role": "assistant", "content": report})
                return jsonify({
                    "ok": False,
                    "message": "자동 실행 중 LLM 호출에 실패했습니다.",
                    "status": status,
                    "game_finished": game_finished,
                    "game_finished_message": game_finished_message,
                    "report": report,
                    "move_log": move_log,
                    "failed_log": failed_log,
                    "auto_trace": auto_trace,
                })

            cmd = extract_command(answer)
            if not cmd:
                append_failed_log("auto", "없음", status, "LLM 응답에서 Command를 찾지 못했습니다.", answer)
                continue

            # 같은 상태에서 이미 탈락한 Command는 다시 실행하지 않는다.
            already_failed = any(
                item["state_key"] == state_key and item["command"] == cmd
                for item in failed_log
            )
            if already_failed:
                append_failed_log("auto", cmd, status, "같은 상태에서 이미 탈락한 Command를 다시 제안했습니다.", answer)
                continue

            before = status.copy()
            ok, after, end, message = validate_candidate(cmd, before)
            if not ok:
                append_failed_log("auto", cmd, before, message, answer)
                continue

            status = after
            log_item = append_move_log("auto", cmd, before, status, message, end)
            executed.append(log_item)
            moved = True

            if end:
                game_finished = True
                game_finished_message = message
            break

        if game_finished:
            break

        if not moved:
            append_failed_log("auto", "중단", status, "현재 상태에서 LLM이 유효한 새 선택지를 찾지 못해 자동 실행을 중단했습니다.")
            break

    if not game_finished and len(executed) >= max_steps:
        append_failed_log("auto", "중단", status, f"최대 자동 실행 턴 수({max_steps})에 도달했습니다.")

    report = build_game_report(short=False)
    history.append({"role": "assistant", "content": report})

    return jsonify({
        "ok": game_finished and "승리" in game_finished_message,
        "message": game_finished_message or "자동 실행을 중단했습니다. 보고서를 확인하세요.",
        "status": status,
        "game_finished": game_finished,
        "game_finished_message": game_finished_message,
        "report": report,
        "move_log": move_log,
        "failed_log": failed_log,
        "auto_trace": auto_trace,
    })


@app.get("/api/history")
def api_history():
    return jsonify({"history": history, "move_log": move_log, "failed_log": failed_log, "auto_trace": auto_trace})


@app.get("/api/report")
def api_report():
    return jsonify({
        "report": build_game_report(short=False),
        "move_log": move_log,
        "failed_log": failed_log,
        "auto_trace": auto_trace,
        "status": status,
        "game_finished": game_finished,
        "game_finished_message": game_finished_message,
    })


@app.post("/api/guide")
def api_guide():
    global history
    body = request.get_json(force=True, silent=True) or {}
    user_text = body.get("message", "").strip()

    if not user_text:
        return jsonify({"answer": "질문을 입력하세요.", "command": None})

    if game_finished:
        answer = (game_finished_message or "게임이 종료되었습니다.") + "\n초기화 후 다시 진행하세요.\n\n" + build_game_report(short=False)
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": answer})
        return jsonify({
            "answer": answer,
            "command": None,
            "history": history[-10:],
            "status": status,
            "game_finished": True,
            "game_finished_message": game_finished_message,
        })

    if is_injection(user_text):
        answer = "이 챗봇은 식인종-선교사 게임 전용입니다. 규칙 변경이나 시스템 지시 공개 요청은 처리하지 않습니다."
        return jsonify({"answer": answer, "command": None, "status": status, "game_finished": game_finished})

    if not is_game_question(user_text):
        answer = "이 챗봇은 식인종-선교사 게임 전용입니다. 게임 상태, 이동 Command, 규칙, 다음 수에 대해서만 질문해 주세요."
        return jsonify({"answer": answer, "command": None, "status": status, "game_finished": game_finished})

    try:
        answer = call_ollama(build_guide_messages(user_text), timeout=120)
    except Exception as e:
        answer = (
            "Ollama 연결 오류입니다. main.py의 OLLAMA_CHAT_URL과 MODEL 값을 확인해 주세요.\n"
            f"현재 OLLAMA_CHAT_URL: {OLLAMA_CHAT_URL}\n"
            f"현재 MODEL: {MODEL}\n"
            f"오류: {e}"
        )

    command = extract_command(answer)

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": answer})

    return jsonify({
        "answer": answer,
        "command": command,
        "history": history[-10:],
        "status": status,
        "game_finished": game_finished,
        "game_finished_message": game_finished_message,
        "move_log": move_log,
        "failed_log": failed_log,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
