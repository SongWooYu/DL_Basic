# Python 3.12.13
# 식인종-선교사 게임 최소 구현
# 구조: while True -> print(status) -> input(cmd) -> game(status, cmd) -> status 갱신
# 시작 방향: 오른쪽 -> 왼쪽

TOTAL_M = 3
TOTAL_C = 3

# Command 정의: (선교사 수, 식인종 수)
COMMANDS = {
    "1M": (1, 0),
    "2M": (2, 0),
    "1C": (0, 1),
    "2C": (0, 2),
    "1M1C": (1, 1),
}


def make_status(left_m, left_c, right_m, right_c, boat):
    """상태 정의"""
    return {
        "left_m": left_m,
        "left_c": left_c,
        "right_m": right_m,
        "right_c": right_c,
        "boat": boat,  # "left" 또는 "right"
    }


def print_status(status):
    """현재 상태 출력"""
    print()
    print("현재 상태")
    print(f"왼쪽  - 선교사: {status['left_m']}, 식인종: {status['left_c']}")
    print(f"오른쪽 - 선교사: {status['right_m']}, 식인종: {status['right_c']}")
    print(f"배 위치: {status['boat']}")
    print("사용 가능 Command: 1M, 2M, 1C, 2C, 1M1C")


def normalize_command(command):
    """대소문자와 공백을 무시하고 Command를 표준 형태로 변환"""
    return command.strip().upper().replace(" ", "")


def is_safe(missionary, cannibal):
    """선교사가 0명이면 안전, 아니면 선교사 수가 식인종 수 이상이어야 안전"""
    return missionary == 0 or missionary >= cannibal


def judge(status):
    """다음 상태 판정: 계속 진행 / 게임오버 / 승리"""
    # 사람 수 범위 확인
    values = [
        status["left_m"], status["left_c"],
        status["right_m"], status["right_c"],
    ]
    if any(v < 0 for v in values):
        return True, "게임오버: 사람 수가 음수가 되었습니다."

    if status["left_m"] + status["right_m"] != TOTAL_M:
        return True, "게임오버: 선교사 전체 수가 맞지 않습니다."

    if status["left_c"] + status["right_c"] != TOTAL_C:
        return True, "게임오버: 식인종 전체 수가 맞지 않습니다."

    # 식인종이 선교사보다 많으면 게임오버
    if not is_safe(status["left_m"], status["left_c"]):
        return True, "게임오버: 왼쪽에서 식인종 수가 선교사 수보다 많습니다."

    if not is_safe(status["right_m"], status["right_c"]):
        return True, "게임오버: 오른쪽에서 식인종 수가 선교사 수보다 많습니다."

    # 승리 조건: 모두 왼쪽으로 이동
    if status["left_m"] == TOTAL_M and status["left_c"] == TOTAL_C:
        return True, "승리: 모두 왼쪽으로 이동했습니다."

    return False, "계속 진행"


def game(status, command):
    """현재 상태와 Command를 받아서 다음 상태를 만든다."""
    command = normalize_command(command)

    if command not in COMMANDS:
        print("잘못된 Command입니다.")
        return status, False

    move_m, move_c = COMMANDS[command]

    # 원본 상태를 직접 바꾸지 않기 위해 복사
    next_status = status.copy()

    if status["boat"] == "right":
        # 오른쪽 -> 왼쪽
        if status["right_m"] < move_m or status["right_c"] < move_c:
            print("해당 사람 수 이상이어야 이동 가능합니다.")
            return status, False

        next_status["right_m"] -= move_m
        next_status["right_c"] -= move_c
        next_status["left_m"] += move_m
        next_status["left_c"] += move_c
        next_status["boat"] = "left"

    else:
        # 왼쪽 -> 오른쪽
        if status["left_m"] < move_m or status["left_c"] < move_c:
            print("해당 사람 수 이상이어야 이동 가능합니다.")
            return status, False

        next_status["left_m"] -= move_m
        next_status["left_c"] -= move_c
        next_status["right_m"] += move_m
        next_status["right_c"] += move_c
        next_status["boat"] = "right"

    end, message = judge(next_status)
    print(message)
    return next_status, end


# 최초 상태: 모두 오른쪽에 있고 배도 오른쪽에 있음
status = make_status(0, 0, 3, 3, "right")

while True:
    print_status(status)
    cmd = input("Command 입력: ")

    status, end = game(status, cmd)

    if end:
        print_status(status)
        break
