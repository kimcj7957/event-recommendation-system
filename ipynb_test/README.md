# 📚 모델 학습 데모 - ipynb_test

이 디렉토리는 이벤트 추천 시스템의 모델 학습 과정을 단계별로 시연하기 위한 파일들을 포함합니다.

## 📁 파일 구성

- **`model_training_demo.ipynb`**: 전체 모델 학습 과정을 보여주는 Jupyter 노트북
- **`simple_model_demo.py`**: 핵심 내용을 Python 스크립트로 구현한 버전
- **`requirements.txt`**: Jupyter 노트북 실행에 필요한 의존성
- **`README.md`**: 이 파일

## 🚀 실행 방법

### 1. Jupyter 노트북 실행

#### 환경 설정
```bash
# ipynb_test 디렉토리로 이동
cd ipynb_test

# 가상환경 활성화 (상위 디렉토리의 .venv 사용)
source ../.venv/bin/activate

# Jupyter 관련 패키지 설치
pip install jupyter notebook matplotlib seaborn
```

#### 노트북 실행
```bash
# Jupyter 노트북 실행
jupyter notebook model_training_demo.ipynb

# 또는 Jupyter Lab 사용
pip install jupyterlab
jupyter lab model_training_demo.ipynb
```

### 2. Python 스크립트 실행

```bash
# 가상환경 활성화
source ../.venv/bin/activate

# 스크립트 실행
python simple_model_demo.py
```

## 📖 학습 내용

### Jupyter 노트북 (`model_training_demo.ipynb`)
1. **데이터 로드 및 탐색**: 이벤트 데이터셋 분석
2. **데이터 전처리**: 텍스트 정제 및 코퍼스 생성
3. **TF-IDF 모델**: 키워드 기반 벡터화 및 KNN 학습
4. **LSA 모델**: 잠재 의미 분석을 통한 차원 축소
5. **Word2Vec 대안**: HashingVectorizer를 이용한 임베딩
6. **하이브리드 모델**: 여러 모델의 조합
7. **성능 비교**: 각 모델의 추천 성능 분석
8. **결과 시연**: 실제 쿼리를 통한 추천 결과

### Python 스크립트 (`simple_model_demo.py`)
- 노트북의 핵심 내용을 간소화한 버전
- 명령행에서 빠르게 실행 가능
- TF-IDF와 LSA 모델 학습 및 테스트

## 🔧 시스템 요구사항

### 필수 패키지
```
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
jupyter >= 1.0.0
scipy >= 1.7.0
joblib >= 1.1.0
```

### 선택사항
```
gensim >= 4.0.0  # 완전한 Word2Vec 구현 (설치 어려움)
```

## 📊 예상 결과

### 학습 완료 후 출력 예시:
```
✅ 데이터 로드 완료: 1,234개 이벤트
🔤 TF-IDF 모델 학습 중...
✅ TF-IDF 모델 완료: (1234, 5023)
🧮 LSA 모델 학습 중...
✅ LSA 모델 완료: (1234, 523)
✅ 총 2개 모델 학습 완료!

🎯 추천 시스템 테스트
테스트 쿼리: {'keywords': '재즈 콘서트', 'price_max': 50000, 'location': '강남구'}

🤖 TFIDF 모델 결과:
1. 📍 블루노트 서울 (강남구)
   💰 45,000원
   📊 유사도: 0.823
   📝 재즈의 전설들이 펼치는 감동적인 라이브 콘서트...
```

## 🎯 학습 목표

이 데모를 통해 다음을 이해할 수 있습니다:

1. **텍스트 전처리**: 실제 데이터의 정제 과정
2. **벡터화 기법**: TF-IDF, Count, Hashing 벡터화 비교
3. **차원 축소**: SVD를 이용한 LSA 구현
4. **유사도 검색**: KNN을 이용한 코사인 유사도 기반 추천
5. **모델 평가**: 실제 쿼리를 통한 성능 비교
6. **하이브리드 접근**: 여러 모델의 조합 방법

## ⚠️ 주의사항

1. **데이터 의존성**: 상위 디렉토리의 `model/recommender_ko.joblib` 파일이 필요합니다.
2. **메모리 사용량**: 전체 데이터셋을 메모리에 로드하므로 충분한 RAM이 필요합니다.
3. **실행 시간**: 모델 학습에 몇 분이 소요될 수 있습니다.

## 🔗 관련 파일

- **백엔드**: `../backend/multi_recommender.py` - 실제 서비스용 구현
- **모델**: `../model/recommender_ko.joblib` - 사전 학습된 모델
- **프론트엔드**: `../frontend/` - 웹 인터페이스

## 💡 다음 단계

1. 노트북에서 다양한 파라미터 실험
2. 새로운 쿼리로 추천 결과 테스트
3. 모델 성능 개선 방법 탐구
4. 실제 웹 서비스에서 결과 확인

---

🎵 **즐거운 머신러닝 학습 되세요!** 🎵 