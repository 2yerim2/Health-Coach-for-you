import os
import json
from pathlib import Path
from openai import OpenAI
from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
import bs4
import numpy as np

# 설정
OUTPUT_JSON = "health_knowledge_embeddings.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "text-embedding-3-small"

# PDF 파일 경로 리스트(위에서부터 순서대로 수면1, 스트레스2, 신체활동2)
PDF_FILES = ["adult-sleep-duration-health-advisory.pdf",
             "9789241505406_eng.pdf",
             "9789240003910-eng.pdf",
             "9789241599979_eng.pdf",
             "Physical_Activity_Guidelines_2nd_edition.pdf",
             ]

# OpenAI 클라이언트 초기화
import os
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("[WARNING] OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    print("[INFO] 환경 변수를 설정하거나 .env 파일을 사용하세요.")
    client = None
else:
    print(f"[INFO] OpenAI API 키 확인됨 (길이: {len(api_key)}자)")
    print(f"[INFO] API 키 시작 부분: {api_key[:20]}...")
    
    # API 키 유효성 간단 테스트
    try:
        test_client = OpenAI(api_key=api_key)
        # 실제 호출은 하지 않고 클라이언트만 생성
        client = test_client
        print("[INFO] OpenAI 클라이언트 초기화 완료")
    except Exception as e:
        print(f"[ERROR] OpenAI 클라이언트 초기화 실패: {e}")
        print("[WARNING] API 키가 유효하지 않을 수 있습니다. 환경 변수를 확인해주세요.")
        client = None

# Flask 앱 초기화
app = Flask(__name__)


def load_pdfs(pdf_paths):
    """
    여러 개의 PDF 파일을 읽어서 LangChain Document 리스트로 합침
    p.resolve(): 절대 경로를 반환
    d.metadata["source"]: 각 문서에 출처(파일 경로) 정보를 메타데이터로 저장
    """
    all_docs = []
    for path in pdf_paths:
        p = Path(path)
        if not p.exists():
            print(f"파일을 찾을 수 없습니다: {p.resolve()}")
            continue
        
        print(f"[INFO] PDF 로딩 중: {p}")
        loader = PyMuPDFLoader(str(p))
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = str(p)  # 각 문서에 출처 정보 저장
            d.metadata["category"] = "sleep"          # 수면 관련
            d.metadata["category"] = "physical_activity"
            d.metadata["category"] = "mental_health"

        all_docs.extend(docs)  # 루프 밖으로 이동하여 중복 방지
            
    return all_docs

def chunk_text(text,chunk_size=CHUNK_SIZE,overlap=CHUNK_OVERLAP):#긴 텍스트를 chunk_size 기준으로 겹치게 잘라서 리스트로 반환.
    chunks=[]
    start=0
    text_length = len(text)
    
    while start < text_length:
        end= start+ chunk_size
        chunk=text[start:end]
        chunks.append(chunk)
        start = end - overlap # overlap만큼 뒤로 덜 가서 다음 chunk 시작
        
    return chunks

def build_chunks_from_docs(docs):
    """
    LangChain Document 리스트를 청크로 분할
    doc.metadata: 문서의 메타데이터 딕셔너리 (source, page 등)
    doc.page_content: 문서의 실제 텍스트 내용
    """
    chunks = []
    chunk_id = 0
    
    for doc in docs:
        source = doc.metadata.get("source", "unknown_source")
        page = doc.metadata.get("page", None) 
        
        raw_text = doc.page_content
        if not raw_text:
            continue
        
        text_chunks = chunk_text(raw_text)
        
        for t in text_chunks:
            if not t.strip():  # 빈 텍스트 제외
                continue
            
            chunks.append({
                'id': f'chunk-{chunk_id}',
                'text': t,
                'source': source,
                'page': page,
            })
            chunk_id += 1
        
    print(f"[INFO] 생성된 chunk 개수: {len(chunks)}")
    return chunks

def embed_chunks(chunks):
    """
    chunk 리스트에 대해 OpenAI 임베딩을 생성하고, 
    각 chunk에 'embedding' 필드를 추가한 리스트를 반환
    """
    embedded = []
    total = len(chunks)
    
    for i, ch in enumerate(chunks, start=1):
        text = ch['text']
        
        print(f"[INFO] 임베딩 생성 중: {i}/{total}")
        resp = client.embeddings.create(  
            model=EMBEDDING_MODEL,
            input=text,
        )
        embedding = resp.data[0].embedding
        
        item = {
            'id': ch['id'],
            'text': ch['text'],
            'source': ch['source'],
            'page': ch['page'],
            'embedding': embedding,  # float 리스트
        }
        embedded.append(item)
            
    return embedded
    
def save_to_json(data, filename):
    """데이터를 JSON 파일로 저장"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] JSON 파일 저장 완료: {filename}")

def build_rag_knowledge_base():
    """RAG 기반 지식 베이스 구축 (임베딩 및 JSON 형식으로 저장)"""
    #PDF를 document로 변환
    docs = load_pdfs(PDF_FILES)
    
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
        print(f"[WARNING] 웹 문서 로딩 실패: {e}")
        
    try:
        print("[INFO] 웹 문서 로딩 중...")
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
    
    print("[INFO] RAG 기반 지식 베이스 구축 완료!")
    return embedded_chunks

KB_JSON_PATH = Path(__file__).with_name("health_knowledge_embeddings.json")

def load_kb(json_path:Path):
    if not json_path.exists():
        raise FileNotFoundError(f'지식베이스 JSON을 찾을 수 없습니다:{json_path}')
    
    with open(json_path,'r',encoding = 'utf-8') as f:
        data = json.load(f)
        
    print(f'[INFO] KB 로드 완료, chunk 수: {len(data)}')
    return data

KB = load_kb(KB_JSON_PATH)

#코사인 유사도 계산
def cosine_sim(vec_a, vec_b):
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a,b) / denom)

# 질문 임베딩 만들고, KB에서 유사도 높은 chunk top_k개 반환
def retrieve_context(question : str, top_k: int = 3):
    try:
        if not KB or len(KB) == 0:
            print("[ERROR] 지식베이스가 비어있습니다.")
            return []
        
        try:
            emb_resp = client.embeddings.create(
                model = "text-embedding-3-small",
                input = question,
            )
        except Exception as emb_error:
            error_msg = str(emb_error)
            print(f"[ERROR] 임베딩 생성 실패: {error_msg}")
            if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                raise ValueError("OpenAI API 키가 유효하지 않습니다. API 키를 확인해주세요.")
            elif "rate_limit" in error_msg.lower():
                raise ValueError("API 사용량 한도에 도달했습니다. 잠시 후 다시 시도해주세요.")
            else:
                raise
        
        if not emb_resp or not emb_resp.data or len(emb_resp.data) == 0:
            print("[ERROR] 임베딩 생성 실패: 응답 데이터가 없습니다")
            return []
        
        q_emb = emb_resp.data[0].embedding
        
        scored = []
        for i in KB:
            if 'embedding' not in i:
                continue
            sim = cosine_sim(q_emb, i['embedding'])
            scored.append((sim, i))
        
        if len(scored) == 0:
            print("[WARNING] 유사한 청크를 찾지 못했습니다.")
            return []
        
        scored.sort(key = lambda x: x[0], reverse = True) # 유사도 내림차순 정렬
        top_items = [x[1] for x in scored[:top_k]] # scored 리스트의 요소인 튜플에서 문서(chunk)를 남김. -> 리스트로 반환
        return top_items
    except Exception as e:
        print(f"[ERROR] 컨텍스트 검색 중 오류: {e}")
        import traceback
        traceback.print_exc()
        raise

def build_prompt(question: str, contexts: list[str], health_data: dict = None) -> str:
    """
    컨텍스트+질문+건강 기록 -> 모델에 넘겨줄 프롬프트 작성
    health_data: 사용자의 건강 기록 요약 정보 (선택사항)
    """
    context_text = "\n\n".join(contexts)
    
    # 건강 기록 정보 포맷팅
    health_info = ""
    if health_data:
        health_info = "\n[사용자 건강 기록]\n"
        health_info += f"- 총 기록 일수: {health_data.get('totalDays', 0)}일\n"  # 딕셔너리에서 key가 없는 경우를 위해 0 출력하도록 만듦
        health_info += f"- 최근 7일간 기록: {health_data.get('recentDays', 0)}일\n"
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

def run_rag_pipeline(question: str, health_data: dict = None) -> str:
    """
    RAG 파이프라인 실행
    question: 사용자 질문
    health_data: 사용자 건강 기록 요약 (선택사항)
    """
    try:
        #KB에서 관련 문맥 검색
        top_chunks = retrieve_context(question, top_k=3)
        
        if not top_chunks or len(top_chunks) == 0:
            print("[WARNING] 검색된 청크가 없습니다. 기본 답변을 생성합니다.")
            # 기본 답변 생성
            return "죄송합니다. 관련 정보를 찾지 못했습니다. 다시 질문해주세요."
    except Exception as e:
        print(f"[ERROR] 컨텍스트 검색 실패: {e}")
        import traceback
        traceback.print_exc()
        raise
    
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
    if client is None:
        raise ValueError("OpenAI 클라이언트가 초기화되지 않았습니다. API 키를 확인해주세요.")
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # 올바른 모델 이름으로 수정
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] OpenAI API 호출 실패: {error_msg}")
        import traceback
        traceback.print_exc()
        
        # 구체적인 에러 메시지 제공
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower() or "invalid" in error_msg.lower():
            raise ValueError("OpenAI API 키가 유효하지 않거나 설정되지 않았습니다. API 키를 확인해주세요.")
        elif "rate_limit" in error_msg.lower() or "quota" in error_msg.lower():
            raise ValueError("API 사용량 한도에 도달했습니다. 잠시 후 다시 시도해주세요.")
        elif "insufficient_quota" in error_msg.lower():
            raise ValueError("API 크레딧이 부족합니다. OpenAI 계정에서 크레딧을 충전해주세요.")
        else:
            raise ValueError(f"OpenAI API 호출 실패: {error_msg}")
    
    # 5) 응답 텍스트 추출
    if not resp or not resp.choices or len(resp.choices) == 0:
        raise ValueError("OpenAI API에서 응답을 받지 못했습니다.")
    
    answer = resp.choices[0].message.content
    if not answer:
        raise ValueError("응답 내용이 비어있습니다.")
    
    return answer
            
    
@app.route("/", methods=["GET"])
def index():
    """메인 페이지: 질문 폼 + 결과 표시"""
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    """질문을 제출받아 RAG 파이프라인 실행 후 결과 렌더링"""
    question = request.form.get("question", "").strip()

    if not question:
        return render_template(
            "index.html",
            error="질문을 입력해 주세요.",
        )

    # 건강 기록 데이터는 폼 요청에서는 받지 않음 (AJAX 요청에서만 사용)
    try:
        answer = run_rag_pipeline(question)
    except Exception as e:
        print("[ERROR] RAG 파이프라인 오류:", e)
        return render_template(
            "index.html",
            error="답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        )
   
    return render_template(
        "index.html",
        question=question,
        answer=answer,
    )

@app.route("/api/gpt-question", methods=["POST"])
def api_gpt_question():
    """
    프론트엔드에서 AJAX(fetch)로 보내는 JSON 요청을 처리하는 엔드포인트.
    요청 형식: { "question": "...", "healthData": {...} } (healthData는 선택사항)
    응답 형식: { "answer": "..." }
    """
    data = request.get_json(silent=True)  # JSON 파싱
    if not data or "question" not in data:
        return jsonify({"error": "질문이 전달되지 않았습니다."}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "질문이 비어 있습니다."}), 400

    # 건강 기록 데이터 가져오기 (선택사항)
    health_data = data.get("healthData", None)
    
    try:
        print(f"[INFO] 질문 수신: {question[:50]}...")
        if health_data:
            print(f"[INFO] 건강 기록 데이터 포함: {health_data.get('totalDays', 0)}일 기록")
        
        answer = run_rag_pipeline(question, health_data)
        print(f"[INFO] 답변 생성 완료: {len(answer)}자")
        
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] RAG 처리 중 오류: {error_msg}")
        import traceback
        traceback.print_exc()
        
        # 사용자 친화적인 에러 메시지
        if "API" in error_msg or "OpenAI" in error_msg:
            user_error = "OpenAI API 연결에 문제가 발생했습니다. API 키를 확인해주세요."
        elif "KB" in error_msg or "지식베이스" in error_msg:
            user_error = "지식베이스를 불러오는 중 오류가 발생했습니다."
        else:
            user_error = f"답변 생성 중 오류가 발생했습니다: {error_msg[:100]}"
        
        return jsonify({"error": user_error}), 500

    return jsonify({"answer": answer})

if __name__ == "__main__":
    # 개발용 서버 실행
    app.run(debug=True, port=8000)
    

