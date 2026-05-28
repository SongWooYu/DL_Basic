from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "다음 텍스트를 핵심 bullet 5개로 한국어 요약해줘."),
    ("human", "텍스트:\n{content}")
    # ("human", "다음 텍스트를 핵심 bullet 5개로 한국어 요약해줘.\n\n텍스트:\n{content}")
])

llm = ChatOllama(
    model = "gemma3:4b",
    temperature = 0.2,
    # base_url = "http://host.docker.internal:11434"
    base_url = "http://192.168.24.184:11434"
)

chain = PROMPT | llm | StrOutputParser()

def summarize(text: str) -> str:
    return chain.invoke({"content": text})

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("사용법: python summarize_chain.py /workspace/input.txt")
        raise SystemExit(1)
    
    text_path = Path(sys.argv[1])
    text = text_path.read_text(encoding="utf-8")

    # print(summarize(text))
    for chunk in chain.stream({"content": text}):
        print(chunk, end="", flush=True)

