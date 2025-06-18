# 📦 Jupyter 노트북 의존성 패키지 가이드

## 🎯 `balanced_training_demo.ipynb` 실행을 위한 완전한 패키지 리스트

### 🚀 빠른 설치 (최소 필수)
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib jupyter notebook
```

### 🔥 완전 설치 (모든 기능)
```bash
pip install -r complete_requirements.txt
```

---

## 📋 패키지별 상세 설명

### 🔴 **필수 패키지** (없으면 노트북 실행 불가)

#### 📚 **Jupyter 환경**
```
jupyter>=1.0.0          # Jupyter 메인 패키지
notebook>=6.0.0         # Jupyter Notebook 인터페이스
ipykernel>=6.0.0        # Python 커널
ipywidgets>=7.6.0       # 인터랙티브 위젯
```

#### 📊 **데이터 처리**
```
pandas>=1.3.0           # ✅ 사용: df = pd.read_csv(), pd.DataFrame()
numpy>=1.21.0           # ✅ 사용: np.array(), np.argsort(), 수치 계산
```

#### 📈 **시각화**
```
matplotlib>=3.5.0       # ✅ 사용: plt.figure(), plt.show()
seaborn>=0.11.0         # ✅ 사용: sns.set_style(), 통계 플롯
```

#### 🤖 **머신러닝**
```
scikit-learn>=1.0.0     # ✅ 사용:
                        # - TfidfVectorizer: TF-IDF 벡터화
                        # - CountVectorizer: 단어 빈도 벡터화
                        # - HashingVectorizer: 해시 벡터화
                        # - TruncatedSVD: LSA 차원 축소
                        # - NearestNeighbors: k-NN 검색
                        # - cosine_similarity: 코사인 유사도
                        # - MinMaxScaler: 데이터 정규화
                        # - OneHotEncoder: 원-핫 인코딩
                        # - ColumnTransformer: 전처리 파이프라인
```

#### 🔬 **과학 계산**
```
scipy>=1.7.0            # ✅ 사용:
                        # - sparse.hstack: 희소 행렬 수평 결합
                        # - sparse.csr_matrix: 압축 희소 행렬
```

#### 💾 **모델 저장/로드**
```
joblib>=1.1.0           # ✅ 사용: joblib.dump(), joblib.load()
```

---

### 🟡 **선택적 패키지** (특정 기능용)

#### 🌐 **웹 API 개발**
```
fastapi>=0.68.0         # 🔧 용도: REST API 서버 구축
pydantic>=1.8.0         # 🔧 용도: 데이터 검증 및 직렬화
uvicorn>=0.15.0         # 🔧 용도: ASGI 웹 서버
```

#### 🔧 **고급 NLP**
```
# gensim>=4.0.0         # 🔧 용도: Word2Vec (설치 어려우면 생략 가능)
```

---

## 📝 **내장 모듈** (별도 설치 불필요)

```python
import re               # 정규표현식
import warnings         # 경고 메시지 제어
import traceback        # 오류 추적
from pathlib import Path # 파일 경로 처리
```

---

## ⚡ **설치 순서 및 주의사항**

### 1️⃣ **가상환경 생성 (권장)**
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# 또는
.venv\Scripts\activate     # Windows
```

### 2️⃣ **필수 패키지 설치**
```bash
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib
```

### 3️⃣ **Jupyter 설치**
```bash
pip install jupyter notebook ipykernel ipywidgets
```

### 4️⃣ **선택적 패키지 설치**
```bash
# API 서버 기능이 필요한 경우
pip install fastapi pydantic uvicorn

# Word2Vec 기능이 필요한 경우 (선택사항)
pip install gensim
```

---

## 🔍 **버전 호환성**

| 패키지 | 최소 버전 | 권장 버전 | 주요 기능 |
|--------|----------|----------|----------|
| pandas | 1.3.0 | 2.0.0+ | 데이터프레임 처리 |
| numpy | 1.21.0 | 1.24.0+ | 수치 계산 |
| scikit-learn | 1.0.0 | 1.3.0+ | 머신러닝 |
| scipy | 1.7.0 | 1.10.0+ | 희소 행렬 |
| matplotlib | 3.5.0 | 3.7.0+ | 시각화 |
| seaborn | 0.11.0 | 0.12.0+ | 통계 플롯 |

---

## 🚨 **트러블슈팅**

### ❌ **일반적인 오류들**

1. **`ModuleNotFoundError: No module named 'sklearn'`**
   ```bash
   pip install scikit-learn
   ```

2. **`ImportError: cannot import name 'sparse' from 'scipy'`**
   ```bash
   pip install --upgrade scipy
   ```

3. **Jupyter 커널 인식 안됨**
   ```bash
   python -m ipykernel install --user --name=.venv
   ```

4. **패키지 버전 충돌**
   ```bash
   pip install --upgrade --force-reinstall scikit-learn
   ```

---

## ✅ **설치 확인**

노트북 첫 번째 셀에서 실행:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import joblib

print("✅ 모든 필수 패키지가 성공적으로 설치되었습니다!")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
``` 