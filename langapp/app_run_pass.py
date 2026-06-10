from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOllama(model="gemma3:4b", temperature=0.2, base_url="http://host.docker.internal:11434")
to_str = StrOutputParser()
summary_prompt = ChatPromptTemplate.from_template(
"다음 텍스트를 한 문장으로 한국어 요약:\n\n{content}"
)
translate_prompt = ChatPromptTemplate.from_template(
"이 문장을 자연스러운 영어로 번역:\n\n{summary}"
)
summary_chain = summary_prompt | llm | to_str
translate_chain = translate_prompt | llm | to_str

pipeline = (
{"summary": summary_chain} # content를 받아 summary 생성
| RunnablePassthrough.assign(translated=translate_chain) # summary를 이용해 번역 실행
)
if __name__ == "__main__":
    text = "LangChain은 체인을 모듈식으로 구성해 LLM 활용을 단순화한다."
    out = pipeline.invoke({"content": text})
    print(out)