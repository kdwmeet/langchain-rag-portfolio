import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
#rag_pipeline파일의 get_rag_chain함수
from rag_pipeline import get_rag_chain

st.set_page_config(page_title="Python 기술 문서 봇")
st.title("Python 기술 문서 봇")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chain" not in st.session_state:
    st.session_state.chain = get_rag_chain()


#대화 기록
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)
input_text = st.chat_input("질문을 입력하세요...")

if input_text:
    with st.chat_message("user"):
        st.markdown(input_text)
    
    with st.chat_message("assistant"):
        with st.spinner("LCEL 파이프라인이 동작 중..."):

            #LCEL 체인 실행
            response = st.session_state.chain.invoke({
                "input": input_text,
                "chat_history": st.session_state.chat_history
            })

            answer = response['answer']
            docs = response['context']

            #출처 정리
            sources = [f"p.{doc.metadata.get('page', '?')}" for doc in docs]
            unique_sources = sorted(list(set(sources)))
            source_text = ", ".join(unique_sources)

            full_response = f"{answer}\n\n===\n** 참고 페이지 {source_text}"
            st.markdown(full_response)

    #기록 저장        
    st.session_state.chat_history.extend([
        HumanMessage(content=input_text),
        AIMessage(content=full_response)
    ])