import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from operator import itemgetter

# .env 로딩 시도
load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    #LLM & DB 설정
    llm = ChatOpenAI(
        model="gpt-5-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'}
    )

    #로컬 DB 로드
    vectorstore = FAISS.load_local(
        "./vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever()

    #질문 재구성 (과거 대화 흐름 파악)
    contextualize_q_syste_prompt = """
    채팅 기록과 최신 질문이 주어지면, 질문을 문맥에 맞게 다시 완전한 문장으로 고쳐주세요.
    답변은 하지 말고 질문만 재구성하세요.
    """
    contextualize_q_syste_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_syste_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    #질문 재구성 체인
    history_aware_retriever = contextualize_q_syste_prompt | llm | StrOutputParser()

    #문서 검색 및 답변 생성 
    qa_system_prompt = """
    당신은 'IT기술 멘토'입니다.
    아래의 Context를 참조하여 질문에 답변하세요.

    규칙:
    1. 초보자도 이해하기 쉽게 설명하세요.
    2. 문서에 코드가 있다면 반드시 코드 블록으로 보여주세요.
    3. 모르면 모른다고 하세요.
    4. 답변 끝에 반드시 출처 페이지를 표기하세요.

    Context: {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    rag_chain = (
        RunnableParallel(
            # 문맥 고려해 검색수행하기
            context=(history_aware_retriever | retriever),
            input=itemgetter("input"),
            chat_history=itemgetter("chat_history")
        )
       .assign(answer=(
            # AI한테 줄 때 문서와 질문을 'format_docs'로 합쳐서 AI에게 보냄
            RunnablePassthrough.assign(
                context=(lambda x: format_docs(x["context"]))
            )
            | qa_prompt
            | llm
            | StrOutputParser()
        ))
        .pick(["answer", "context"]) # 결과에는 최종답변과 근거문서만 반환
    )

    return rag_chain