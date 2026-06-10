from collections import defaultdict
from typing import Dict

from flask import Flask
from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

app = Flask(__name__)
    
llm = ChatOllama(
    model = "gemma3:4b",
    temperature = 0.7,
    # base_url = "http://host.docker.internal:11434"
    base_url = "http://192.168.24.184:11434")
to_str = StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "너는 친절한 한국어 비서야. 대화 맥락을 잘 이어가."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)
base_chain = prompt | llm | to_str

_STORE : Dict[str, InMemoryChatMessageHistory] = defaultdict(InMemoryChatMessageHistory)

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    return _STORE[session_id]


chat_chain = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

# if __name__ == "__main__":
#     sid = "user-1"
#     print(chat_chain.invoke(
#         {"input" : "안녕? 오늘 할일 정리해줘"},
#         config={"configurable" : {"session_id" : sid}}
#     ))
#     print(chat_chain.invoke(
#         {"input" : "내가 방금 뭐라 했지?"},
#         config={"configurable" : {"session_id" : sid}}
#     ))




if __name__ == "__main__":
    sid = "user-1"

    while True:
        text = input("입력: ")

        if text == "/exit" or text == "x" or text == "X":
            break

        result = chat_chain.invoke(
            {"input": text},
            config={"configurable": {"session_id": sid}}
        )

        print(result)





























# if __name__ == "__main__":
#     sid = "user-1"

#     print("메모리 챗봇 시작")
#     print("종료하려면 /exit 입력")

#     while True:
#         user_input = input("\nUSER> ").strip()

#         if user_input == "/exit":
#             print("프로그램을 종료합니다.")
#             break

#         if not user_input:
#             continue

#         try:
#             answer = chat_chain.invoke(
#                 {"input": user_input},
#                 config={"configurable": {"session_id": sid}}
#             )
#             print("\nAI>", answer)

#         except KeyboardInterrupt:
#             print("\n프로그램을 종료합니다.")
#             break

#         except Exception as e:
#             print("\nERROR>", e)