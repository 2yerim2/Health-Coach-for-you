# 🏥 건강 관리 웹앱 (Health Management Web App)

RAG(Retrieval-Augmented Generation) 기반 AI 건강 코칭 시스템과 개인 건강 기록 관리 기능을 제공하는 웹 애플리케이션입니다.

## 📋 목차

- [주요 기능](#주요-기능)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [RAG 파이프라인](#rag-파이프라인)
- [주의사항](#주의사항)

## ✨ 주요 기능

### 1. 건강 기록 관리
- **일일 건강 지표 입력**
  - 수면 시간 (시간 단위)
  - 운동 시간 (분 단위)
  - 스트레스 수준 (1-5 척도)
  - 메모 (자유 입력)
- **하루 1회 입력 제한**: 중복 기록 방지
- **데이터 시각화**: Chart.js를 활용한 누적 건강 추이 그래프
- **로컬 저장**: 브라우저 localStorage 기반 데이터 관리

### 2. RAG 기반 AI 건강 코칭
- **신뢰할 수 있는 근거 기반 답변**
  - WHO, CDC 등 공신력 있는 건강 자료 활용
  - 출처 표시 (문서명, 페이지 번호)
- **맞춤형 건강 조언**
  - 사용자 질문에 대한 최적 컨텍스트 검색
  - 실천 가능한 구체적 행동 제안
- **안전 기능**
  - 의료 진단 금지
  - 생활습관 개선 중심 조언
  - 면책 문구 자동 포함

## 🛠 기술 스택

### Backend
- **Flask**: Python 웹 프레임워크
- **OpenAI API**: GPT-4, Embeddings 모델
- **LangChain**: 문서 로딩 및 처리
- **NumPy**: 벡터 연산 및 유사도 계산

### Frontend
- **HTML/CSS/JavaScript**: 기본 웹 기술
- **Chart.js**: 데이터 시각화
- **AJAX (Fetch API)**: 비동기 서버 통신

### 데이터 처리
- **PyMuPDF**: PDF 문서 파싱
- **BeautifulSoup**: 웹 크롤링
- **JSON**: 데이터 저장 형식

## 📁 프로젝트 구조

```
ex6/
├── app.py                          # Flask 메인 애플리케이션
├── health_knowledge_embeddings.json # RAG 지식 베이스 (임베딩 데이터)
├── adult-sleep-duration-health-advisory.pdf
├── 9789241505406_eng.pdf           # WHO 건강 문서들
├── 9789240003910-eng.pdf
├── 9789241599979_eng.pdf
├── Physical_Activity_Guidelines_2nd_edition.pdf
├── templates/
│   └── index.html                  # 메인 페이지 템플릿
└── static/
    └── style.css                   # 스타일시트
```

## 🚀 설치 방법

### 1. 필수 요구사항
- Python 3.8 이상
- OpenAI API 키

### 2. 패키지 설치

```bash
pip install flask
pip install openai
pip install langchain-community
pip install beautifulsoup4
pip install numpy
pip install pymupdf
```

또는 requirements.txt가 있다면:

```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정

OpenAI API 키를 환경 변수로 설정:

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

또는 `.env` 파일 생성:
```
OPENAI_API_KEY=your-api-key-here
```

### 4. 지식 베이스 구축 (선택사항)

기존 `health_knowledge_embeddings.json` 파일이 없다면, `app.py`에서 다음 함수를 실행하여 생성:

```python
if __name__ == "__main__":
    build_rag_knowledge_base()  # 주석 해제
    app.run(debug=True, port=5001)
```

## 📖 사용 방법

### 1. 애플리케이션 실행

```bash
python app.py
```

서버가 `http://localhost:5001`에서 실행됩니다.

### 2. 건강 기록 입력

1. 브라우저에서 `http://localhost:5001` 접속
2. "오늘의 건강 기록" 섹션에서:
   - 수면 시간 입력
   - 운동 시간 입력
   - 스트레스 수준 선택 (1-5)
   - 메모 작성 (선택사항)
3. "기록 저장" 버튼 클릭
4. 하루에 한 번만 입력 가능

### 3. 건강 추이 확인

- "오늘의 건강 코치" 섹션에서 그래프 확인
- 수면 시간, 운동 시간, 스트레스 수준의 누적 추이 시각화
- 처음 입력한 날짜부터 모든 기록 표시

### 4. AI 건강 상담

1. "AI 건강 상담" 섹션에서 질문 입력
2. "질문하기" 버튼 클릭
3. RAG 파이프라인을 통해 근거 기반 답변 수신
4. 답변에는 출처 정보 포함

## 🔄 RAG 파이프라인

### 1. 지식 베이스 구축

```
PDF/웹 문서 → 텍스트 추출 → 청킹 (500자, 100자 오버랩) 
→ 임베딩 생성 → JSON 저장
```

### 2. 질문 처리 과정

```
사용자 질문 
  ↓
질문 임베딩 생성
  ↓
코사인 유사도로 상위 3개 청크 검색
  ↓
검색된 컨텍스트 + 질문 → GPT-4 프롬프트
  ↓
근거 기반 답변 생성
  ↓
출처 정보와 함께 사용자에게 제공
```

### 3. 지식 베이스 구성

**PDF 문서:**
- 수면 관련: adult-sleep-duration-health-advisory.pdf
- 정신건강 관련: WHO 문서들
- 신체활동 관련: Physical_Activity_Guidelines_2nd_edition.pdf

**웹 문서:**
- CDC 수면 정보
- CDC 정신건강 정보
- WHO 신체활동 정보
- CDC 신체활동 이점

## ⚠️ 주의사항

### 의료 진단 금지
- 이 서비스는 **의료 진단이 아닌 생활습관 가이드**입니다
- 심각한 건강 문제가 있다면 반드시 전문의와 상담하세요
- AI의 답변은 참고용이며, 의료 조언을 대체할 수 없습니다

### 데이터 저장
- 건강 기록은 **브라우저 localStorage**에 저장됩니다
- 브라우저 캐시 삭제 시 데이터가 손실될 수 있습니다
- 중요한 데이터는 별도로 백업하세요

### API 비용
- OpenAI API 사용 시 비용이 발생합니다
- 임베딩 생성 및 GPT-4 호출에 따라 비용이 달라집니다
- 개발/테스트 시 사용량을 모니터링하세요

### 지식 베이스 업데이트
- 새로운 건강 가이드라인이 발표되면 지식 베이스를 업데이트해야 합니다
- `build_rag_knowledge_base()` 함수를 실행하여 재구축하세요

## 🔧 주요 함수 설명

### `load_pdfs(pdf_paths)`
여러 PDF 파일을 LangChain Document 리스트로 변환

### `chunk_text(text, chunk_size, overlap)`
긴 텍스트를 지정된 크기로 청킹 (오버랩 포함)

### `build_chunks_from_docs(docs)`
Document 리스트를 청크로 분할하고 메타데이터 추가

### `embed_chunks(chunks)`
청크 리스트에 OpenAI 임베딩 생성

### `retrieve_context(question, top_k)`
질문과 유사한 상위 k개 청크 검색

### `run_rag_pipeline(question)`
전체 RAG 파이프라인 실행 (검색 → 프롬프트 → 답변)

## 📝 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

## 👤 작성자

건강 관리 웹앱 프로젝트

## 🙏 참고 자료

- [OpenAI API 문서](https://platform.openai.com/docs)
- [LangChain 문서](https://python.langchain.com/)
- [Flask 문서](https://flask.palletsprojects.com/)
- WHO 건강 가이드라인
- CDC 건강 정보

---

**면책 조항**: 이 서비스는 의료 진단이 아닌 생활습관 가이드입니다. 건강 문제가 있다면 전문의와 상담하세요.

