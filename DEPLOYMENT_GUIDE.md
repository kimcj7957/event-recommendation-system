# 📚 GitHub 업로드 및 배포 가이드

이 가이드는 이벤트 추천 시스템을 GitHub에 업로드하고 배포하는 방법을 설명합니다.

## 📋 준비사항

1. **GitHub 계정** (https://github.com)
2. **Git 설치** (`git --version`으로 확인)
3. **GitHub CLI (선택사항)**: `brew install gh` (macOS)

## 🚀 GitHub 업로드 단계

### 1. Git 저장소 초기화

```bash
# 프로젝트 루트 디렉토리에서 실행
cd data_mining

# Git 저장소 초기화
git init

# 사용자 정보 설정 (한 번만)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 2. 파일 추가 및 커밋

```bash
# 모든 파일 스테이징
git add .

# 첫 번째 커밋
git commit -m "Initial commit: Multi-model Event Recommendation System

- FastAPI backend with 4 ML models (TF-IDF, LSA, Word2Vec, Hybrid)
- Next.js frontend with like functionality
- Complete installation and usage documentation"
```

### 3. GitHub 리포지토리 생성

#### 방법 A: GitHub 웹사이트에서
1. https://github.com 접속
2. 우상단 "+" → "New repository" 클릭
3. Repository name: `event-recommendation-system`
4. Description: `Multi-model event recommendation system with TF-IDF, LSA, Word2Vec, and Hybrid models`
5. Public 선택 (또는 Private)
6. "Create repository" 클릭

#### 방법 B: GitHub CLI 사용
```bash
# GitHub CLI로 리포지토리 생성
gh repo create event-recommendation-system --public --description "Multi-model event recommendation system"
```

### 4. 원격 저장소 연결 및 푸시

```bash
# 원격 저장소 추가 (YOUR_USERNAME을 실제 GitHub 사용자명으로 변경)
git remote add origin https://github.com/YOUR_USERNAME/event-recommendation-system.git

# 기본 브랜치를 main으로 설정
git branch -M main

# GitHub에 푸시
git push -u origin main
```

## 🌐 배포 옵션

### 옵션 1: Vercel (프론트엔드) + Railway (백엔드)

#### 프론트엔드 배포 (Vercel)
1. https://vercel.com 접속
2. GitHub 계정으로 로그인
3. "Import Project" → GitHub 리포지토리 선택
4. Root Directory: `frontend` 설정
5. Environment Variables 추가:
   ```
   NEXT_PUBLIC_API_URL=https://your-backend-url.railway.app
   ```

#### 백엔드 배포 (Railway)
1. https://railway.app 접속
2. GitHub 계정으로 로그인
3. "New Project" → "Deploy from GitHub repo"
4. Root Directory: `backend` 설정
5. Environment Variables 추가:
   ```
   PORT=8000
   PYTHON_VERSION=3.11
   ```

### 옵션 2: Heroku (풀스택)

#### Heroku 설정 파일 생성

1. **Procfile** (루트 디렉토리):
```
web: cd backend && python main.py
```

2. **runtime.txt** (루트 디렉토리):
```
python-3.11.10
```

3. **package.json** (루트 디렉토리):
```json
{
  "name": "event-recommendation-system",
  "version": "1.0.0",
  "scripts": {
    "build": "cd frontend && npm install && npm run build",
    "start": "cd backend && python main.py"
  }
}
```

#### Heroku 배포
```bash
# Heroku CLI 설치 및 로그인
brew install heroku/brew/heroku
heroku login

# 앱 생성
heroku create your-app-name

# 환경변수 설정
heroku config:set PYTHON_VERSION=3.11

# 배포
git push heroku main
```

### 옵션 3: Docker (로컬/클라우드)

#### Dockerfile.backend
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

COPY backend/ .
COPY model/ ./model/

CMD ["python", "main.py"]
```

#### Dockerfile.frontend
```dockerfile
FROM node:18-alpine

WORKDIR /app
COPY frontend/package*.json ./
RUN npm install

COPY frontend/ .
RUN npm run build

CMD ["npm", "start"]
```

#### docker-compose.yml
```yaml
version: '3.8'
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      - PORT=8000

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - backend
```

## 🔧 환경변수 설정

### 백엔드 (.env)
```bash
# backend/.env
PORT=8000
MODEL_PATH=../model/recommender_ko.joblib
```

### 프론트엔드 (.env.local)
```bash
# frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 📝 커밋 메시지 규칙

```bash
# 기능 추가
git commit -m "feat: add new recommendation model"

# 버그 수정
git commit -m "fix: resolve CORS issue in API"

# 문서 업데이트
git commit -m "docs: update installation guide"

# 스타일 변경
git commit -m "style: improve UI responsiveness"

# 리팩토링
git commit -m "refactor: optimize model loading performance"
```

## 🚨 주의사항

1. **모델 파일**: 100MB 미만이므로 Git에 포함 가능
2. **API 키**: `.env` 파일은 `.gitignore`에 포함됨
3. **의존성**: `requirements.txt`와 `package.json` 최신 상태 유지
4. **CORS**: 프로덕션에서는 적절한 CORS 설정 필요

## 📊 모니터링

배포 후 다음 URL들을 확인:
- **프론트엔드**: https://your-app.vercel.app
- **백엔드 API**: https://your-api.railway.app/docs
- **헬스 체크**: https://your-api.railway.app/health

---

🚀 **성공적인 배포를 위해 단계별로 진행하세요!** 