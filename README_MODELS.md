# 🎵 다중 모델 이벤트 추천 시스템

## 📋 프로젝트 개요

본 프로젝트는 **TF-IDF, LSA(Latent Semantic Analysis), Word2Vec, Hybrid** 등 다양한 머신러닝 모델을 활용한 이벤트 추천 시스템입니다. 각 모델의 고유한 특성을 살려 다양한 관점에서 이벤트를 추천하며, 사용자가 원하는 추천 방식을 선택할 수 있도록 구현되었습니다.

## 🎯 프로젝트 목표

- **기존 TF-IDF 시스템의 한계 극복**: 단순한 키워드 매칭을 넘어선 의미적 유사성 고려
- **다중 모델 비교**: 각 모델의 성능과 특성을 실시간으로 비교 가능
- **사용자 선택권 제공**: 상황에 맞는 최적의 추천 모델 선택
- **확장 가능한 아키텍처**: 새로운 모델 추가가 용이한 구조

## 🤖 구현된 모델들

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)

#### 📌 특징
- **빠른 처리 속도**: 실시간 검색에 최적화
- **정확한 키워드 매칭**: 입력된 키워드와 정확히 일치하는 이벤트 탐지
- **해석 가능성**: 결과에 대한 명확한 설명 가능

#### 🔧 구현 방식
```python
# TF-IDF 벡터화
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    stop_words='english'
)
X_text = tfidf.fit_transform(text_corpus)

# 메타데이터와 결합
X_combined = hstack([X_text, X_meta]).tocsr()

# KNN 모델 학습
knn = NearestNeighbors(metric='cosine', n_neighbors=20, n_jobs=-1)
knn.fit(X_combined)
```

#### ✅ 장점
- 구현이 간단하고 안정적
- 메모리 효율적
- 키워드 기반 검색에 매우 효과적

#### ⚠️ 한계
- 동의어나 유사한 의미의 단어 처리 어려움
- 문맥 정보 미고려
- 어휘 범위를 벗어난 검색에 취약

### 2. LSA (Latent Semantic Analysis)

#### 📌 특징
- **잠재 의미 분석**: SVD를 통한 차원 축소로 의미적 유사성 탐지
- **동의어 처리**: 표면적으로 다른 단어들 간의 의미적 연관성 파악
- **노이즈 감소**: 차원 축소를 통한 노이즈 제거 효과

#### 🔧 구현 방식
```python
# Count 벡터화 (LSA에 더 적합)
count_vec = CountVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    stop_words='english'
)
X_text = count_vec.fit_transform(text_corpus)

# SVD로 차원 축소 (잠재 의미 분석)
svd = TruncatedSVD(n_components=100, random_state=42)
X_text_reduced = svd.fit_transform(X_text)

# 메타데이터와 결합
X_combined = np.hstack([X_text_reduced, X_meta])
```

#### ✅ 장점
- 의미적 유사성 고려
- 동의어 및 관련 개념 처리 가능
- 차원 축소로 계산 효율성 향상

#### ⚠️ 한계
- 해석이 어려움 (블랙박스 특성)
- 매개변수 튜닝 필요
- 작은 데이터셋에서 성능 저하 가능

### 3. Word2Vec

#### 📌 특징
- **단어 임베딩**: 단어 간 의미적 관계를 벡터 공간에 표현
- **문맥 고려**: 주변 단어들과의 관계를 학습
- **의미적 연산**: 단어 간 유사도 및 관계 계산 가능

#### 🔧 구현 방식
```python
# 텍스트 전처리 및 토큰화
sentences = [simple_preprocess(text) for text in text_corpus]

# Word2Vec 모델 학습
w2v_model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    epochs=10,
    sg=1  # Skip-gram
)

# 문서 벡터 생성 (단어 벡터들의 평균)
def get_document_vector(tokens, model):
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
```

#### ✅ 장점
- 풍부한 의미 정보 포함
- 단어 간 관계 학습
- 전이 학습 가능

#### ⚠️ 한계
- 훈련 시간이 오래 걸림
- 대용량 메모리 필요
- OOV(Out-of-Vocabulary) 문제

### 4. Hybrid Model

#### 📌 특징
- **다중 모델 조합**: 여러 모델의 장점을 결합
- **균형잡힌 결과**: 각 모델의 편향성 완화
- **신뢰성 향상**: 다양한 관점에서의 검증

#### 🔧 구현 방식
```python
def _recommend_hybrid(self, query: dict, top_k: int = 5):
    component_models = self.models['hybrid']['component_models']
    all_recommendations = []
    
    # 각 모델에서 추천 결과 수집
    for model_name in component_models:
        recs = self.recommend(query, model_name, top_k * 2)
        recs['model_source'] = model_name
        all_recommendations.append(recs)
    
    # 결과 조합 및 재순위
    combined = pd.concat(all_recommendations, ignore_index=True)
    
    # 중복 제거 및 점수 조합
    unique_results = self._remove_duplicates_and_combine_scores(combined)
    
    return result_df.sort_values('score', ascending=False).head(top_k)
```

#### ✅ 장점
- 각 모델의 강점 활용
- 더 균형잡힌 추천 결과
- 모델별 편향성 감소

#### ⚠️ 한계
- 계산 비용 증가
- 복잡한 매개변수 조정
- 모델 간 가중치 설정 필요

## 🏗️ 시스템 아키텍처

### Backend 구조
```
backend/
├── main.py                 # FastAPI 메인 애플리케이션
├── multi_recommender.py    # 다중 모델 추천 시스템
├── recommender.py          # 기존 단일 TF-IDF 모델 (폴백용)
├── requirements.txt        # 의존성 패키지
└── model/
    └── recommender_ko.joblib # 사전 훈련된 모델 파일
```

### Frontend 구조
```
frontend/src/app/
├── page.tsx                # 메인 페이지
├── components/
│   ├── SearchForm.tsx      # 검색 폼 (모델 선택 포함)
│   └── RecommendationResults.tsx # 추천 결과 표시
└── globals.css            # 전역 스타일
```

### API 엔드포인트

#### 1. `/models` - 사용 가능한 모델 조회
```http
GET /models
```
**응답:**
```json
{
  "available_models": ["tfidf", "lsa", "word2vec", "hybrid"],
  "model_descriptions": {
    "tfidf": "TF-IDF: 키워드 빈도 기반 추천 (빠름, 정확한 키워드 매칭)",
    "lsa": "LSA: 잠재 의미 분석 기반 추천 (의미적 유사성 고려)",
    "word2vec": "Word2Vec: 단어 임베딩 기반 추천 (단어 간 의미 관계 고려)",
    "hybrid": "Hybrid: 여러 모델 조합 추천 (종합적 결과)"
  }
}
```

#### 2. `/recommend` - 이벤트 추천
```http
POST /recommend
```
**요청 본문:**
```json
{
  "keywords": "재즈 콘서트",
  "price_max": 50000,
  "location": "서울 마포구",
  "model": "lsa",
  "top_k": 5
}
```

#### 3. `/recommend/compare` - 모델 비교
```http
POST /recommend/compare
```

## 📊 모델 성능 비교

| 모델 | 처리 속도 | 의미 이해 | 키워드 정확도 | 메모리 사용량 | 적용 상황 |
|------|-----------|-----------|---------------|---------------|-----------|
| **TF-IDF** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 정확한 키워드 검색 |
| **LSA** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 의미적 유사성 중시 |
| **Word2Vec** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | 단어 관계 중시 |
| **Hybrid** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | 종합적 추천 |

## 🔧 기술적 구현 세부사항

### 1. 데이터 전처리
- **텍스트 정규화**: 소문자 변환, 특수문자 제거
- **토큰화**: 공백 및 구두점 기준 분리
- **불용어 제거**: 의미 없는 단어 필터링
- **메타데이터 인코딩**: 가격, 지역 정보 수치화

### 2. 벡터화 과정
```python
# 텍스트 코퍼스 생성
text_corpus = (
    df['content'].fillna('') + ' ' +
    df['place'].fillna('') + ' ' +
    df['loc_sigu'].fillna('')
)

# 메타데이터 전처리
meta_features = ['price_adv', 'price_door', 'loc_sigu']
meta_preprocessor = ColumnTransformer([
    ('price_scaler', MinMaxScaler(), ['price_adv', 'price_door']),
    ('location_encoder', OneHotEncoder(handle_unknown='ignore'), ['loc_sigu'])
])
```

### 3. 유사도 계산
- **코사인 유사도**: 벡터 간 각도 기반 유사성 측정
- **KNN (K-Nearest Neighbors)**: 가장 유사한 k개 이벤트 탐색
- **거리 메트릭**: 코사인 거리 사용으로 정규화 효과

### 4. 에러 처리 및 폴백
```python
try:
    # 다중 모델 시스템 사용
    recommendations_df = multi_recommender.recommend(query_dict, model_name, top_k)
except Exception as e:
    # 단일 모델 폴백
    recommendations_df = single_recommend(query_dict, top_k)
    print(f"Fallback to single model: {e}")
```

## 🚀 시작하기

### 1. 환경 설정
```bash
# 백엔드 의존성 설치
cd backend
pip install -r requirements.txt

# 프론트엔드 의존성 설치
cd frontend
npm install
```

### 2. 서버 실행
```bash
# 백엔드 서버 시작 (http://localhost:8000)
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 프론트엔드 개발 서버 시작 (http://localhost:3000)
cd frontend
npm run dev
```

### 3. 사용법
1. 브라우저에서 `http://localhost:3000` 접속
2. 검색 키워드, 가격, 지역 입력
3. 원하는 추천 모델 선택
4. 추천 결과 확인

## 📈 성능 최적화

### 1. 메모리 최적화
- **희소 행렬(Sparse Matrix) 사용**: 메모리 효율성 향상
- **지연 로딩**: 필요시에만 모델 로드
- **배치 처리**: 대량 요청 처리 최적화

### 2. 처리 속도 개선
- **병렬 처리**: 멀티코어 활용 (`n_jobs=-1`)
- **캐싱**: 자주 사용되는 결과 캐시 저장
- **인덱싱**: 효율적인 검색을 위한 인덱스 구축

### 3. 모델 경량화
- **차원 축소**: PCA, SVD를 통한 특성 수 감소
- **어휘 제한**: 빈도 기반 어휘 필터링
- **양자화**: 모델 크기 압축

## 🔍 문제 해결 과정

### 1. 기존 TF-IDF 시스템의 한계 분석
**문제점:**
- 높은 유사도 점수에도 불구하고 관련 없는 이벤트 추천
- 키워드 기반 매칭의 한계
- 의미적 유사성 미고려

**해결 방안:**
- 다중 모델 도입으로 다양한 관점에서 추천
- LSA를 통한 잠재 의미 분석
- Word2Vec을 통한 단어 관계 학습

### 2. 기술적 도전과 해결

#### 2.1 NumPy 배열 호환성 문제
**문제:** `'numpy.ndarray' object has no attribute 'toarray'`
**해결:**
```python
# sparse matrix 여부 확인 후 처리
if hasattr(X_meta, 'toarray'):
    X_meta = X_meta.toarray()
```

#### 2.2 모델 초기화 실패 처리
**문제:** 일부 모델 초기화 실패시 전체 시스템 중단
**해결:**
```python
try:
    self.build_word2vec_model()
except Exception as e:
    print(f"Word2Vec initialization failed: {e}")
    # 다른 모델들은 계속 사용 가능
```

#### 2.3 메모리 효율성 개선
**문제:** 대용량 벡터 데이터로 인한 메모리 부족
**해결:**
- 희소 행렬 활용
- 배치 처리 구현
- 불필요한 데이터 즉시 해제

## 🎯 향후 개선 계획

### 1. 모델 확장
- **BERT/Transformer 기반 모델**: 더 정교한 언어 이해
- **딥러닝 기반 추천**: Neural Collaborative Filtering
- **앙상블 모델**: 여러 모델의 가중 조합

### 2. 개인화 기능
- **사용자 프로필**: 개인 취향 학습
- **협업 필터링**: 유사 사용자 기반 추천
- **피드백 학습**: 사용자 반응 기반 모델 업데이트

### 3. 실시간 처리
- **스트리밍 데이터**: 실시간 이벤트 업데이트
- **온라인 학습**: 새로운 데이터로 모델 지속 업데이트
- **A/B 테스팅**: 모델 성능 실시간 비교

### 4. UI/UX 개선
- **시각화**: 추천 근거 시각적 표현
- **필터링**: 고급 검색 옵션
- **모바일 최적화**: 반응형 디자인

## 📚 참고 자료

### 논문 및 이론
- [TF-IDF: A Study of the Term Frequency Inverse Document Frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis)
- [Word2Vec: Efficient Estimation of Word Representations](https://arxiv.org/abs/1301.3781)

### 기술 문서
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Gensim Word2Vec Tutorial](https://radimrehurek.com/gensim/models/word2vec.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)

### 라이브러리
- **scikit-learn**: 머신러닝 알고리즘
- **gensim**: Word2Vec 구현
- **FastAPI**: 백엔드 API 프레임워크
- **Next.js**: 프론트엔드 프레임워크
- **NumPy/Pandas**: 데이터 처리

## 🏆 프로젝트 성과

### 기술적 성과
- ✅ **4개 모델 성공적 구현**: TF-IDF, LSA, Word2Vec, Hybrid
- ✅ **확장 가능한 아키텍처**: 새로운 모델 추가 용이
- ✅ **실시간 모델 비교**: 사용자가 직접 성능 체험 가능
- ✅ **안정적인 폴백 시스템**: 오류 발생시 대체 모델 자동 활용

### 사용자 경험 개선
- ✅ **직관적 UI**: 모델 선택과 결과 비교가 쉬운 인터페이스
- ✅ **실시간 피드백**: 즉시 추천 결과 확인 가능
- ✅ **다양한 선택권**: 상황에 맞는 최적 모델 선택
- ✅ **서울 지역 특화**: 사용자 요구에 맞춘 지역 제한

---

**💡 이 프로젝트는 단순한 키워드 매칭을 넘어 의미 이해와 다양한 관점을 통한 지능형 추천 시스템의 가능성을 보여줍니다.** 