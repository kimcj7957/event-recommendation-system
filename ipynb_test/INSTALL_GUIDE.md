# 🚀 노트북 실행을 위한 간단 설치 가이드

## ⚡ **1분 설치** (가장 중요!)

```bash
# 1. 가상환경 생성 (권장)
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# 2. 필수 패키지 설치 (한 번에!)
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib jupyter notebook ipykernel

# 3. Jupyter 실행
jupyter notebook
```

## 📦 **핵심 패키지 리스트**

### 🔴 **절대 필수** (이것들만 있으면 노트북 실행 가능)
```
pandas          # 데이터 처리
numpy           # 수치 계산  
scikit-learn    # 머신러닝
scipy           # 희소 행렬
matplotlib      # 플롯
seaborn         # 시각화
joblib          # 모델 저장
jupyter         # 노트북 환경
notebook        # Jupyter 인터페이스
```

### 🟡 **선택사항** (고급 기능용)
```
fastapi         # API 서버 (필요시)
pydantic        # 데이터 검증 (필요시)
uvicorn         # 웹 서버 (필요시)
gensim          # Word2Vec (어려우면 생략)
```

## ✅ **설치 확인**

노트북에서 이 코드 실행:
```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
print("✅ 설치 완료!")
```

## 🆘 **문제 해결**

1. **`No module named 'sklearn'` 오류**
   ```bash
   pip install scikit-learn
   ```

2. **가상환경이 인식 안될 때**
   ```bash
   python -m ipykernel install --user --name=.venv
   ```

3. **모든 패키지 다시 설치**
   ```bash
   pip install --upgrade --force-reinstall pandas numpy scikit-learn scipy
   ```

---

**🎯 핵심**: 위의 **1분 설치** 명령어만 실행하면 노트북이 완벽하게 작동합니다! 