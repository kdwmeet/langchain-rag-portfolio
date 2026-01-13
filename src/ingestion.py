from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_docs():

    print("'manual.pdf' 문서 불러오는중...")
    loader = PyPDFLoader("./data/manual.pdf")
    raw_docs = loader.load()

    #슬라이싱으로 필요한 부분만 남김
    START_PAGE = 8
    END_PAGE = -6

    docs = raw_docs[START_PAGE:END_PAGE]
    print(f"전처리 완료 : 총 {len(raw_docs)}페이지 중 불필요한 앞뒤를 자르고 {len(docs)}페이지만 학습합니다.")

    #문서 분할, 문맥보존 위해 200자 겹치게 자르기 
    print("문서를 자르는 중...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    #벡터 DB 저장, 한국어 문장 유사도 측정에 뛰어난 모델 사용
    print("데이터 벡터화하여 저장 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'}
    )

    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local("./vectorstore")

    print("'vectorstore' 폴더 생성 완료")

if __name__ == "__main__":
    ingest_docs()
