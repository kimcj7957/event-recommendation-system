# 🎵 이벤트 추천 시스템

다중 모델 기반 이벤트 추천 시스템입니다. TF-IDF, LSA, Word2Vec, Hybrid 모델을 지원하여 사용자의 취향에 맞는 이벤트를 추천합니다.

## 🚀 주요 기능

- **4가지 추천 모델**: TF-IDF, LSA, Word2Vec (대안 구현), Hybrid
- **실시간 검색**: 키워드, 가격, 지역 기반 필터링
- **좋아요 기능**: 관심 이벤트 표시 및 관리
- **모던 UI**: Next.js 기반 반응형 웹 인터페이스
- **FastAPI 백엔드**: 빠르고 안정적인 API 서버

## 📋 시스템 요구사항

### 필수 소프트웨어
- **Python**: 3.9 이상 (3.13 권장)
- **Node.js**: 18.0 이상
- **npm**: 8.0 이상

### 운영체제
- **macOS**: 10.15 이상
- **Windows**: 10 이상
- **Linux**: Ubuntu 18.04 이상

## 🛠️ 설치 및 실행 가이드

### 1. 프로젝트 압축 해제
```bash
# 다운로드 받은 zip 파일 압축 해제
unzip data_mining.zip
cd data_mining
```

### 2. 백엔드 설정 및 실행

#### Python 가상환경 생성 및 활성화
```bash
# 가상환경 생성
python3 -m venv .venv

# 가상환경 활성화
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
```

#### 의존성 설치
```bash
# Python 패키지 설치
pip install --upgrade pip
pip install fastapi "uvicorn[standard]" pydantic pandas numpy scikit-learn joblib
```

#### 백엔드 서버 실행
```bash
cd backend
python main.py
```

✅ **성공 메시지**: 
```
🚀 Multi-Model Event Recommendation API Starting Up...
✅ Available models: ['tfidf', 'lsa', 'word2vec', 'hybrid']
INFO: Uvicorn running on http://0.0.0.0:8000
```

### 3. 프론트엔드 설정 및 실행

#### 새 터미널 창에서 실행
```bash
cd frontend

# 의존성 설치
npm install

# npm도 install 하셔야 합니다!
brew install node
brew install next react react-dom


# 개발 서버 실행
npm run dev
```

✅ **성공 메시지**:
```
▲ Next.js 15.3.3
- Local:        http://localhost:3000
- Network:      http://192.168.x.x:3000
```

### 4. 웹 브라우저에서 접속
- **프론트엔드**: http://localhost:3000
- **API 문서**: http://localhost:8000/docs

## 🎯 사용 방법

### 이벤트 검색
1. 웹 브라우저에서 `http://localhost:3000` 접속
2. 키워드 입력 (예: "재즈", "록 콘서트", "클래식")
3. 옵션 설정:
   - **최대 가격**: 원하는 가격 범위
   - **지역**: 서울 지역 선택
   - **추천 모델**: 4가지 모델 중 선택
   - **결과 개수**: 3~20개

### 추천 모델 설명
- **TF-IDF**: 키워드 빈도 기반 (빠름, 정확한 키워드 매칭)
- **LSA**: 잠재 의미 분석 기반 (의미적 유사성 고려)
- **Word2Vec**: 단어 임베딩 기반 (단어 간 의미 관계 고려)
- **Hybrid**: 여러 모델 조합 (종합적 결과)

### 좋아요 기능
- 각 이벤트 카드의 ❤️ 버튼 클릭
- 관심 이벤트로 표시 및 관리

## 🔧 문제 해결

### 자주 발생하는 문제

#### 1. 포트 충돌 오류
```bash
ERROR: [Errno 48] error while attempting to bind on address ('0.0.0.0', 8000): address already in use
```
**해결방법**:
```bash
# 기존 프로세스 종료
pkill -f "python.*main.py"
# 또는 특정 포트 사용 프로세스 종료
lsof -ti:8000 | xargs kill -9
```

#### 2. macOS 보안 경고
```
Apple은 'next-swc.darwin-arm64.node'에 사용자의 Mac에 손상을 입힐 수 있는 악성 코드가 없음을 확인할 수 없습니다.
```
**해결방법**:
```bash
# 터미널에서 실행
sudo xattr -r -d com.apple.quarantine frontend/node_modules/@next/swc-darwin-arm64/next-swc.darwin-arm64.node
```

#### 3. Python 패키지 설치 오류
**해결방법**:
```bash
# 최신 pip로 업그레이드
pip install --upgrade pip setuptools wheel

# 개별 패키지 설치 시도
pip install fastapi
pip install uvicorn[standard]
pip install pandas numpy scikit-learn
```

#### 4. Node.js 의존성 설치 오류
**해결방법**:
```bash
# npm 캐시 정리
npm cache clean --force

# node_modules 삭제 후 재설치
rm -rf node_modules package-lock.json
npm install
```

### gensim 설치 (선택사항)
Word2Vec의 완전한 구현을 원한다면:

#### macOS
```bash
# Homebrew로 필요 도구 설치
brew install gfortran openblas pkg-config

# 환경변수 설정 후 설치
export LDFLAGS="-L/opt/homebrew/opt/openblas/lib"
export CPPFLAGS="-I/opt/homebrew/opt/openblas/include"
pip install gensim
```

## 📁 프로젝트 구조

```
data_mining/
├── backend/                 # FastAPI 백엔드
│   ├── main.py             # 메인 서버 파일
│   ├── multi_recommender.py # 추천 시스템 로직
│   └── requirements.txt    # Python 의존성
├── frontend/               # Next.js 프론트엔드
│   ├── src/app/           # 앱 소스코드
│   ├── package.json       # Node.js 의존성
│   └── tailwind.config.js # UI 스타일링
├── model/                 # 머신러닝 모델
│   └── recommender_ko.joblib
└── README.md              # 이 파일
```

## 🆘 지원 및 문의

프로젝트 실행 중 문제가 발생하면:

1. **포트 확인**: 8000번(백엔드), 3000번(프론트엔드) 포트가 사용 가능한지 확인
2. **가상환경 확인**: Python 가상환경이 활성화되어 있는지 확인
3. **의존성 확인**: 모든 패키지가 올바르게 설치되었는지 확인

## 📝 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

---

🎵 **즐거운 이벤트 추천 서비스를 경험해보세요!** 🎵 