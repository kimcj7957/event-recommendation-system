import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack
import joblib

# 모델 파일 경로
MODEL_PATH = Path(__file__).parent.parent / 'model' / 'recommender_ko.joblib'

# 전역 변수로 모델 로드
_model = None

def load_model():
    """모델을 로드하고 전역 변수에 저장"""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def encode_query(q: dict):
    """사용자 입력(dict) → 모델 입력 벡터"""
    model = load_model()
    
    # 실제 데이터에서 계산된 중간값 사용
    mid_price = 25000.0
    
    # 안전한 값 추출 및 기본값 설정
    keywords = q.get('keywords', '').strip()
    if not keywords:
        keywords = ''  # 빈 문자열로 처리
    
    price_max = q.get('price_max')
    if price_max is None or pd.isna(price_max):
        price_max = mid_price
    else:
        price_max = float(price_max)
    
    location = q.get('location', '').strip()
    if not location:
        location = 'unknown'
    
    # TF-IDF 벡터화
    txt_vec = model['tfidf'].transform([keywords])
    
    # 메타데이터 처리
    meta_df = pd.DataFrame([{
        'price_adv': price_max,
        'price_door': price_max,
        'loc_sigu': location
    }])
    
    # NaN 체크 및 처리
    meta_df = meta_df.fillna({
        'price_adv': mid_price,
        'price_door': mid_price,
        'loc_sigu': 'unknown'
    })
    
    meta_vec = model['pre'].transform(meta_df)
    
    # 최종 벡터 결합
    combined_vec = hstack([txt_vec, meta_vec])
    
    # NaN 체크
    if hasattr(combined_vec, 'data'):
        # sparse matrix인 경우
        if np.any(np.isnan(combined_vec.data)):
            print("Warning: NaN detected in sparse matrix, replacing with 0")
            combined_vec.data = np.nan_to_num(combined_vec.data)
    else:
        # dense matrix인 경우
        if np.any(np.isnan(combined_vec)):
            print("Warning: NaN detected in dense matrix, replacing with 0")
            combined_vec = np.nan_to_num(combined_vec)
    
    return combined_vec

def recommend(query: dict, top_k=5):
    """추천 함수"""
    try:
        model = load_model()
        
        # 쿼리 벡터 생성
        q_vec = encode_query(query)
        
        # KNN 검색
        dist, idx = model['knn'].kneighbors(q_vec, n_neighbors=top_k)
        
        # 결과 생성
        recs = model['df'].iloc[idx[0]].copy()
        recs['score'] = 1 - dist[0]  # 코사인 유사도
        
        # 날짜/시간을 안전하게 문자열로 변환
        recs['date'] = recs['date'].astype(str).replace('NaT', 'unknown')
        recs['time'] = recs['time'].astype(str).replace('NaT', 'unknown')
        
        # NaN 값 처리
        result_cols = ['link','content','place','date','time','price_adv','price_door','score']
        result = recs[result_cols].fillna({
            'link': '',
            'content': '',
            'place': '',
            'date': 'unknown',
            'time': 'unknown',
            'price_adv': 0,
            'price_door': 0,
            'score': 0
        })
        
        return result
        
    except Exception as e:
        print(f"Error in recommend function: {e}")
        raise e 