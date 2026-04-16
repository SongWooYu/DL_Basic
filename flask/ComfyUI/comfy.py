from flask import Flask, render_template, request
import requests, json, time, os, random

COMFY_URL = "http://192.168.24.184:8188"
WORKFLOW_PATH = "./static/test.json"

app = Flask(__name__)

def load_workflow(path: str) -> dict:
    """JSON 워크플로를 파일에서 읽기"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Workflow not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def update_prompts(graph: dict, pos: str, neg: str, pos_node_id: str = "6", neg_node_id: str = "7") -> dict:
    """워크플로 그래프에서 프롬프트 노드 업데이트"""
    graph[pos_node_id]["inputs"]["text"] = pos
    graph[neg_node_id]["inputs"]["text"] = neg
    return graph

def submit_prompt(graph: dict) -> str:
    r = requests.post(f"{COMFY_URL}/prompt", json={"prompt": graph})
    r.raise_for_status()
    return r.json()["prompt_id"]

def poll_history(prompt_id: str, timeout_sec: int = 60, interval: float = 1.0) -> dict | None:
    """
    간단 폴링 : /history/{prompt_id} 에서 완료 결과를 받을 때까지 대기
    """
    end = time.time() + timeout_sec
    while time.time() < end:
        h = requests.get(f"{COMFY_URL}/history/{prompt_id}")
        if h.status_code == 200 and h.json():
            return list(h.json().values())[0]  # 첫 번째 결과 반환
        time.sleep(interval)
    return None

def extract_first_image(history_block: dict) -> str:
    """
    히스토리 블록에서 SaveImage 출력의 첫 번재 이미지를 view URL로 추출
    """
    # if not history_block:
    #     return ""
    # outputs = history_block.get("outputs", {})
    # for node_id, node_out in outputs.items():
    #     if "images" in node_out["images"]:
    #         img = node_out["images"][0]
    #         fn = img.get("filename")
    #         sub = img.get("subfolder", "")
    #         t = img.get("type", "output")
    #         if fn:
    #             return f"{COMFY_URL}/view?filename={fn}&subfolder={sub}&type={t}"
    # return "" 

    if not history_block:
        return ""
    outputs = history_block.get("outputs", {})
    for node_id, node_out in outputs.items():
        # node_out 안에 "images" 키가 있고, 그 값이 비어있지 않은지 확인
        if "images" in node_out and node_out["images"]: 
            img = node_out["images"][0]
            fn = img.get("filename")
            sub = img.get("subfolder", "")
            t = img.get("type", "output")
            if fn:
                return f"{COMFY_URL}/view?filename={fn}&subfolder={sub}&type={t}"
    return ""

def update_workflow(graph: dict, pos: str, neg: str, seed: int) -> dict:
    """워크플로 그래프에서 프롬프트 및 시드값 업데이트"""
    # 긍정/부정 프롬프트 업데이트 (노드 6, 7)
    graph["6"]["inputs"]["text"] = pos
    graph["7"]["inputs"]["text"] = neg
    
    # 시드값 업데이트 
    graph["3"]["inputs"]["seed"] = seed 
    return graph


def get_all_history_images():
    """/history API를 호출하여 모든 생성 기록의 이미지 URL 리스트 반환"""
    try:
        # 전체 히스토리 가져오기 
        r = requests.get(f"{COMFY_URL}/history")
        r.raise_for_status()
        history = r.json()
        
        image_urls = []
        # 최신 순으로 정렬하기 위해 역순으로 탐색하거나 처리할 수 있습니다.
        for prompt_id in history:
            history_block = history[prompt_id]
            # 기존에 만든 이미지 추출 함수 재활용 [cite: 507, 522]
            url = extract_first_image(history_block) 
            if url:
                image_urls.append(url)
        return image_urls
    except Exception as e:
        print(f"Error fetching history: {e}")
        return []



@app.route('/', methods=['GET', 'POST'])
def index():
    pos = request.form.get("pos", "a beautiful landscape with galaxy in a bottle")
    neg = request.form.get("neg", "text, watermark")
    img_url = ""
    history_images = []
    
    show_history = request.args.get('history') == 'true'

    if request.method == 'POST':
        random_seed = random.randint(0, 112589906842624)
        graph = load_workflow(WORKFLOW_PATH)
        graph = update_workflow(graph, pos, neg, random_seed)
        prompt_id = submit_prompt(graph)
        history_block = poll_history(prompt_id)
        img_url = extract_first_image(history_block)

    if show_history:
        history_images = get_all_history_images()

    return render_template('image.html', 
                           pos=pos, neg=neg, 
                           img_url=img_url, 
                           history_images=history_images, 
                           show_history=show_history)

    # return render_template('image.html', pos=pos, neg=neg, img_url=img_url)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)