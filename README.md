1. 프로젝트 개요

목표: 방대한 기술 문서를 일일이 찾지 않고, 질문 한 번으로 정확한 출처(페이지)와 답변을 얻기 위함.

주요 기능:

PDF 문서 업로드 및 벡터 DB 구축 (FAISS)

대화 맥락(Context)을 이해하는 연속 질문 가능

답변의 근거(출처 페이지) 자동 표기 기능 (신뢰성 확보)

2. 사용 기술 (Tech Stack)

Language: Python 3.11

Framework: LangChain, Streamlit

LLM: OpenAI GPT-5-mini

Embedding: jhgan/ko-sroberta-multitask (HuggingFace 한국어 특화 모델)

Vector DB: FAISS (CPU 환경 최적화)

3. 아키텍처 (Architecture)

<img width="739" height="2817" alt="diagram" src="https://github.com/user-attachments/assets/779089ab-f4a5-4338-9766-18ed8f3a7edb" />


4. 트러블 슈팅 (Troubleshooting)

문제: 초기 Intel Arc B580 GPU 기반 로컬 LLM(Ollama) 도입 시도 중 호환성 이슈 및 잦은 크래시 발생.

해결: 프로젝트의 안정성과 답변 품질을 최우선으로 고려하여, 로컬 GPU 집착을 버리고 OpenAI API(GPT-5-mini)와 하이브리드 방식(임베딩은 로컬, 추론은 클라우드)으로 아키텍처 변경.
