from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(
    model = "gemma3:4b",
    temperature = 0.2,
    # base_url = "http://host.docker.internal:11434"
    base_url = "http://192.168.24.184:11434"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise assistant. Reply in Korean."),
    ("human", "한 줄로 자기소개해줘.")
])

chain = prompt | llm | StrOutputParser()

if __name__ == "__main__":
    print(chain.invoke({}))