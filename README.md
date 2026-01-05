# Health-Coach-for-you(당신을 위한 건강 코치 : 개인 맞춤 건강 보조 서비스)
## 제작의도
-삼성의 디지털 헬스 앱에서 영감을 얻음

-의료 진단과 건강 관리가 점점 개인화되고 있는 시대에 효율적이면서도 정확한 근거로 일상의 건강 관리를 도와줄 수 있는 웹앱을 만들고자 함

## 개발 시 문제점과 개선방법
### (1)건강 관련 데이터의 민감성 및 낮은 접근성
#### 문제 
- 처음 구상할 때는 일상 건강이 아닌 전문적인 의료 데이터(당뇨, 고혈압, 심혈관 질환)를 이용하려고 했었음. 그러나 이러한 질병 관련 통계 데이터가 일반인에게 제공되지 않거나 공개되어 있지 않아 근거 데이터로 이용하기에 부적절하고, 개인 정보 및 신뢰도에 있어 비전문가가 이용하기에는 한계가 있었음. 

#### 개선
- 아주 전문적인 지식이 아닌, 일반인에게 공개된 자료들만으로도 건강 상태를 파악하고 코치 해 줄 수 있는 일상적인 건강 지표(수면, 스트레스, 운동량)를 이용하게됨

### (2) 데이터 수집
#### 문제
- 모티브가 된 삼성 헬스 앱에서는 링이나 워치를 이용하여 개인의 건강 데이터를 구체적으로 수집함. 그러나 내가 만든 웹앱 수준에서는 다른 디바이스나 외부 데이터를 직접 연결하여 가져오도록 구현하기에는 한계가 있었음.

#### 개선
- 사용자가 스스로 자가 진단할 수 있는 수준의 건강 상태 체크 지표를 이용하여 직접 입력할 수 있도록 구현함.

### (3) 파인튜닝과 RAG
#### 문제
- GPT를 이용하여 사용자의 건강 관련 질문에 답변하는 기능을 만들때 GPT 모델을 직접 학습시켜서 이용하려고 했었음 . 그러나 새 자료를 계속 반영하고, 근거를 명확히 제시해야 하며, 업데이트가 잦은 건강 코치 앱의 특성상 이러한 파인튜닝은 적절하지 못했음.최신 기록을 반영하려면 모델을 다시 학습시켜야 하고, 시간/비용이 많이 들며, 잘못 학습시킬 경우 되돌리기가 어렵다는 큰 문제가 있었음. 또 할루시네이션(환각)으로 인해 왜곡된 건강 정보가 제공될 위험이 있었음.

#### 개선
- RAG 기법을 이용하여 공신력 있는 자료들에서 사용자가 입력한 질문과 유사한 키워드를 찾아내 근거로 제시할 수 있게 함. 이를 통해 파인 튜닝 기법에 비해 비용과 시간이 덜 들고, 따로 모델을 학습시키는 것이 아니라 최신 자료를 추가하거나 변경해 주면 되는 간단한 방법임.또한 출처 명확히 제시 가능. 항상 문서 기반의 답변을 하기 때문에 할루시네이션 개선 가능

### (4) 개인화
#### 문제
- 건강 기록 데이터를 기록하고 이를 가지고 개인의 건강 추이 그래프를 확인할 수 있었음. 그러나 더 나아가서 이 건강 기록을 가지고 AI 건강코치에게 질문했을 때 사용자 개인의 기록이 아니라, RAG를 이용해 검색할 수 있는 일반적인 정보들에 대해서만 답변함. 즉, 사용자 개인의 건강 기록 정보가 전혀 반영되지 않고 있었음.

#### 개선
- health data 딕셔너리를 문자열로 포맷팅 -> 총 기록 일수, 최근 7일간 기록, 평균 수면 시간, 평균 운동 시간, 평균 스트레스 수준 데이터를 health_info 문자열에 추가 후 오늘의 건강 기록을 가져와 기록하게 함. 이걸 프롬프트에 추가하여 모델이 나의 누적 기록과 오늘 건강 기록을 반영하여 답변할 수 있도록 개선함.

## 웹앱 소개
### 건강 기록 관리 기능
• 일일 건강 지표 입력: 매일 수면 시간, 운동 시간, 스트레스 수준(1-5), 메모를 입력하여 기록할 수 있음

• 하루 1회 입력 제한 -> 중복 기록 방지

• 브라우저 localStorage 기반 누적 기록으로 데이터 저장

• 시각화: Chart.js로 수면/운동/스트레스 추이 그래프 (누적 데이터)

## RAG 기반 AI 건강 코칭 시스템
### 1) 지식 베이스 구축
- PDF 기반 문서 ( 수면 1개, 스트레스 2개, 신체활동2개)/ WHO, CDC 권고안 
- PDF 기반 문서 ( 수면 1개, 스트레스 2개, 신체활동2개)/ WHO, CDC 권고안
- 
-웹 문서 크롤링 (수면1개, 스트레스 1개, 신체활동 2개)/ WHO, CDC 공식 웹사이트

-텍스트 청킹 : 500자 단위, 100자 오버랩

-임베딩 : OpenAI text-embedding-3-small로 청킹 된 단어 벡터화함

### 2)검색 및 답변
-코사인 유사도 -> 상위 3개 청크 검색

-GPT-4 이용-> 컨텍스트(과거의 기록, 사용자 기록 이용) 기반 답변 생성

-출처 표시

### 3)안전 유의 기능
-의료 진단 언급 금지(의료 진단이 아닌 생활 습관 가이드임을 명시함)

-실제 의사들의 전문적인 진단 느낌이 아닌, 생활습관 개선 중심의 조언

-면책 문구 자동 포함하도록 프롬프트 엔지니어링 함

## 기술 스택
• 백엔드: Flask (Python)

• AI/ML: OpenAI API (GPT-4, Embeddings), LangChain

• 데이터 처리: NumPy (유사도 계산), BeautifulSoup (웹 크롤링)

• 프론트엔드: HTML/CSS/JavaScript, Chart.js(그래프)

• 데이터 저장: JSON 파일, localStorage(건강 기록 누적)

## 주요 특징
- WHO,CDC 등의 공신력있는 문서와 공식 홈페이지에 공개된 웹 문서를 이용해 의료, 건강과 관련되어있는 만큼 근거있는 답변을 제시함

-사용자 맞춤으로, 사용자가 입력한 건강 기록들을 누적하고 이를 바탕으로 개인 건강 기록과 연계한 자세하고 구체적인 조언이 가능

-자연어 처리 모델에 넘겨줄 프롬프트를 구체적으로 작성( '일상 건강 습관 코치'라는 페르소나 부여, 반드시 지켜야 할 제약조건 설정, 답변 형식 예시를 제공하는 퓨 샷 기법 사용)하여 신뢰도와 효율성을 보장.

-RAG 기반으로 제공된 문서에서 질문에 대한 근거를 직접 찾고, GPT가 근거를 바탕으로 답변 생성 및 출처를 제공하여 신뢰할 수 있고 검증 가능한 답변 제공

## 코드
### RAG 기반 지식 베이스 구축(웹 문서, PDF 문서)
```
def build_rag_knowledge_base():
    docs = load_pdfs(PDF_FILES)

    #수면
    try:
        print("[INFO] 웹 문서 로딩 중...")
        loader_sleep = WebBaseLoader(
            web_paths=("https://www.cdc.gov/sleep/about/index.html?utm_source=chatgpt.com",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=(
                        "cdc-page-title cdc-page-offset syndicate", 
                        "cdc-page-content cdc-page-offset syndicate container",
                    )
                )
            ),
        )
        docs_sleep = loader_sleep.load()
        
        for d in docs_sleep:
            d.metadata["source"] = "https://www.cdc.gov/sleep/about/index.html"
            d.metadata["category"] = "sleep"
            
        docs.extend(docs_sleep)
        print(f"[INFO] 웹 문서 로딩 완료: {len(docs_sleep)}개 문서")
    except Exception as e:
        print(f"[WARNING] 웹 문서 로딩 실패: {e}")
 
    #스트레스   
    try:
        print("[INFO] 웹 문서 로딩 중... (CDC Mental Health)")
        loader_mental = WebBaseLoader(
            web_paths=("https://www.cdc.gov/mental-health/living-with/index.html",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=(
                        "cdc-page-title cdc-page-offset syndicate",
                        "cdc-page-content cdc-page-offset syndicate container",
                    )
                )
            ),
        )

        docs_mental = loader_mental.load()

        for d in docs_mental:
            d.metadata["source"] = "https://www.cdc.gov/mental-health/living-with/index.html"
            d.metadata["category"] = "mental_health"

        docs.extend(docs_mental)
        print(f"[INFO] 웹 문서 로딩 완료: CDC Mental Health -> {len(docs_mental)}개 문서")

    except Exception as e:
        print(f"[WARNING] 웹 문서 로딩 실패 (CDC Mental Health): {e}")

    #신체 활동
    try:
        print("[INFO] 웹 문서 로딩 중...")
        loader_act = WebBaseLoader(
            web_paths=("https://www.who.int/news-room/fact-sheets/detail/physical-activity",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=(
                        "sf-item-header-wrapper", 
                        "row sf-detail-content",
                    )
                )
            ),
        )
        docs_act = loader_act.load()
        
        for d in docs_act:
            d.metadata["source"] = "https://www.who.int/news-room/fact-sheets/detail/physical-activity"
            d.metadata["category"] = "activity"
            
        docs.extend(docs_act)
        print(f"[INFO] 웹 문서 로딩 완료: {len(docs_act)}개 문서")
    except Exception as e:
        print        print(f"[WARNING] 웹 문서 로딩 실패: {e}

    #신체 활동        print("[INFO] 웹 문서 로딩 중...")
        loader_act2 = WebBaseLoader(
            web_paths=("https://www.cdc.gov/physical-activity-basics/benefits/index.html",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=(
                        "cdc-page-title cdc-page-offset syndicate", 
                        "cdc-page-content cdc-page-offset syndicate container",
                    )
                )
            ),
        )
        docs_act2 = loader_act2.load()
        
        for d in docs_act2:
            d.metadata["source"] = "https://www.cdc.gov/physical-activity-basics/benefits/index.html"
            d.metadata["category"] = "activity"
            
        docs.extend(docs_act2)
        print(f"[INFO] 웹 문서 로딩 완료: {len(docs_act2)}개 문서")
    except Exception as e:
        print(f"[WARNING] 웹 문서 로딩 실패: {e}")

    # 문서를 청크로 분할
    chunks = build_chunks_from_docs(docs)
    
    # 청크에 임베딩 생성
    embedded_chunks = embed_chunks(chunks)
    
    # JSON 파일로 저장
    save_to_json(embedded_chunks, OUTPUT_JSON)
    
    return embedded_chunks
```

### 청크의 코사인 유사도 계산 및 유사도 높은 청크 k개 반환
```
def cosine_sim(vec_a, vec_b):
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a,b) / denom)

# 질문 임베딩 만들고, KB에서 유사도 높은 chunk top_k개 반환
def retrieve_context(question : str, top_k: int = 3):
    emb_resp = client.embeddings.create(
        model = "text-embedding-3-small",
        input = question,
    )    
    q_emb = emb_resp.data[0].embedding
    
    scored = []
    for i in KB:
        sim = cosine_sim(q_emb, i['embedding'])
        scored.append((sim,i))
        
    scored.sort(key = lambda x: x[0], reverse = True) # 유사도 내림차순 정렬
    top_items = [x[1] for x in scored[:top_k]] # scored 리스트의 요소인 튜플에서 문서(chunk)를 남김. -> 리스트로 반환
    return top_items
```

### 모델에 넘길 프롬프트 작성(누적 건강 기록 + 사용자 질문 + 컨텍스트)
```
def build_prompt(question: str, contexts: list[str], health_data: dict = None) -> str:
    context_text = "\n\n".join(contexts)
    
    # 건강 기록 정보 포맷팅
    health_info = ""
    if health_data:
        health_info = "\n[사용자 건강 기록]\n"
        health_info += f"- 총 기록 일수: {health_data.get('totalDays', 0)}일\n" # 딕셔너리에서 key가 없는 경우를 위해 0 출력하도록 만듦.        health_info += f"- 최근 7일간 기록: {health_data.get('recentDays', 0)}일\n"
        health_info += f"- 평균 수면 시간: {health_data.get('averageSleep', 0)}시간\n"
        health_info += f"- 평균 운동 시간: {health_data.get('averageExercise', 0)}분\n"
        health_info += f"- 평균 스트레스 수준: {health_data.get('averageStress', 0)}/5\n"
        
        today = health_data.get('today')
        if today:
            health_info += f"\n[오늘의 기록]\n"
            health_info += f"- 수면 시간: {today.get('sleepTime', 0)}시간\n"
            health_info += f"- 운동 시간: {today.get('exerciseTime', 0)}분\n"
            health_info += f"- 스트레스: {today.get('stress', 0)}/5\n"
        
        latest_memo = health_data.get('latestMemo')
        if latest_memo:
            health_info += f"\n[최근 메모]\n{latest_memo}\n"
        
        health_info += "\n위 건강 기록 정보를 참고하여 사용자에게 개인화된 조언을 제공해주세요.\n"
    
    prompt = f"""
너는 한국 성인을 대상으로 일상 건강 습관을 코칭하는 AI 코치야.
아래 '근거 자료'를 우선적으로 참고해서, 사용자의 질문에 답변해 줘.

단, 다음을 반드시 지켜:
- 의료적 진단이나 병명은 언급하지 말 것
- 생활 습관 개선과 유지에 집중할 것
- 너무 훈계조가 아니라, 현실적으로 실천 가능한 조언을 줄 것
- 사용자의 건강 기록이 제공된 경우, 그 데이터를 바탕으로 개인화된 조언을 제공할 것
- 마지막 줄에는 반드시 "이 서비스는 의료 진단이 아닌 생활습관 가이드입니다."를 포함할 것

[근거 자료]
{context_text}
{health_info}
[사용자 질문]
{question}

[답변 형식 예시]
1) 현재 상태에 대한 간단 평가 (건강 기록이 있다면 이를 참고)
2) 내일/이번 주에 실천해볼 수 있는 행동 2~3가지 (사용자의 현재 상태에 맞춰)
3) 한 문장으로 응원 메시지

"*이 서비스는 의료 진단이 아닌 생활습관 가이드입니다:)"
"""
    return prompt.strip() 
```

### RAG 파이프라인 실행
```
def run_rag_pipeline(question: str, health_data: dict = None) -> str:

    #KB에서 관련 문맥 검색
    top_chunks = retrieve_context(question, top_k=3)
    
    #컨텍스트용 텍스트 리스트 만들기(출처 포함)
    context_blocks = []
    for ch in top_chunks:
        source = ch.get('source','unknown_source')
        page = ch.get('page',None)
        header = f"[출처: {source}"
        if page is not None:
            header += f", page {page}"
        header +="]"
        block = f"{header}\n{ch['text']}" #(출처\n 내용) 이런 형식의 리스트 만들어짐
        context_blocks.append(block)
        
   # 3) 프롬프트 구성 (건강 기록 정보 포함)
    prompt = build_prompt(question, context_blocks, health_data)

# 4) Chat Completions API 호출
    resp = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
)

# 5) 응답 텍스트 추출
    answer = resp.choices[0].message.content
    return answer
```

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


