#!/usr/bin/env python3
"""
🎵 이벤트 추천 시스템 - 간단한 모델 학습 데모

이 스크립트는 Jupyter notebook의 핵심 내용을 Python 스크립트로 구현한 버전입니다.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# 머신러닝 라이브러리
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack
import re

def preprocess_text(text):
    """텍스트 전처리 함수"""
    if pd.isna(text):
        return ''
    text = re.sub(r'[^\w\s]', ' ', str(text).lower())
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_data():
    """기존 모델에서 데이터 로드"""
    model_path = Path('../model/recommender_ko.joblib')
    
    if not model_path.exists():
        print("❌ 모델 파일을 찾을 수 없습니다. 먼저 백엔드를 실행해주세요.")
        return None, None
    
    base_model = joblib.load(model_path)
    df = base_model['df']
    meta_preprocessor = base_model['pre']
    
    print(f"✅ 데이터 로드 완료: {len(df)}개 이벤트")
    return df, meta_preprocessor

def build_tfidf_model(df, meta_preprocessor):
    """TF-IDF 모델 구축"""
    print("🔤 TF-IDF 모델 학습 중...")
    
    # 텍스트 전처리 및 코퍼스 생성
    df['content_clean'] = df['content'].apply(preprocess_text)
    df['place_clean'] = df['place'].apply(preprocess_text)
    df['location_clean'] = df['loc_sigu'].apply(preprocess_text)
    
    text_corpus = (
        df['content_clean'].fillna('') + ' ' +
        df['place_clean'].fillna('') + ' ' +
        df['location_clean'].fillna('')
    )
    
    # TF-IDF 벡터화
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words='english'
    )
    
    X_text = tfidf_vectorizer.fit_transform(text_corpus)
    X_meta = meta_preprocessor.transform(df)
    X_combined = hstack([X_text, X_meta]).tocsr()
    
    # KNN 모델 학습
    knn = NearestNeighbors(metric='cosine', n_neighbors=10, n_jobs=-1)
    knn.fit(X_combined)
    
    print(f"✅ TF-IDF 모델 완료: {X_combined.shape}")
    
    return {
        'vectorizer': tfidf_vectorizer,
        'knn': knn,
        'text_corpus': text_corpus
    }

def build_lsa_model(df, meta_preprocessor, text_corpus):
    """LSA 모델 구축"""
    print("🧮 LSA 모델 학습 중...")
    
    # Count Vectorizer
    count_vectorizer = CountVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words='english'
    )
    
    X_text = count_vectorizer.fit_transform(text_corpus)
    
    # SVD 차원 축소
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_text_reduced = svd.fit_transform(X_text)
    
    # 메타데이터와 결합
    X_meta = meta_preprocessor.transform(df)
    X_meta_dense = X_meta.toarray() if hasattr(X_meta, 'toarray') else X_meta
    X_combined = np.hstack([X_text_reduced, X_meta_dense])
    
    # KNN 모델
    knn = NearestNeighbors(metric='cosine', n_neighbors=10, n_jobs=-1)
    knn.fit(X_combined)
    
    print(f"✅ LSA 모델 완료: {X_combined.shape}")
    
    return {
        'count_vectorizer': count_vectorizer,
        'svd': svd,
        'knn': knn
    }

def test_recommendations(models, df, meta_preprocessor):
    """추천 시스템 테스트"""
    print("\n🎯 추천 시스템 테스트")
    print("=" * 50)
    
    test_query = {
        "keywords": "재즈 콘서트",
        "price_max": 50000,
        "location": "강남구"
    }
    
    print(f"테스트 쿼리: {test_query}")
    
    for model_name, model in models.items():
        print(f"\n🤖 {model_name.upper()} 모델 결과:")
        print("-" * 30)
        
        try:
            # 쿼리 인코딩 (간단 버전)
            keywords = test_query["keywords"]
            
            if model_name == 'tfidf':
                q_vec = model['vectorizer'].transform([keywords])
                # 메타데이터 간단히 처리
                meta_df = pd.DataFrame([{
                    'price_adv': test_query["price_max"],
                    'price_door': test_query["price_max"],
                    'loc_sigu': test_query["location"]
                }])
                meta_vec = meta_preprocessor.transform(meta_df)
                q_combined = hstack([q_vec, meta_vec])
            
            elif model_name == 'lsa':
                count_vec = model['count_vectorizer'].transform([keywords])
                q_reduced = model['svd'].transform(count_vec)
                meta_df = pd.DataFrame([{
                    'price_adv': test_query["price_max"],
                    'price_door': test_query["price_max"],
                    'loc_sigu': test_query["location"]
                }])
                meta_vec = meta_preprocessor.transform(meta_df)
                meta_vec_dense = meta_vec.toarray() if hasattr(meta_vec, 'toarray') else meta_vec
                q_combined = np.hstack([q_reduced, meta_vec_dense])
            
            # 추천 실행
            distances, indices = model['knn'].kneighbors(q_combined, n_neighbors=3)
            similarities = 1 - distances[0]
            
            # 결과 출력
            for i, (idx, sim) in enumerate(zip(indices[0], similarities), 1):
                event = df.iloc[idx]
                print(f"{i}. 📍 {event['place']} ({event.get('loc_sigu', 'N/A')})")
                print(f"   💰 {event.get('price_adv', 0):,.0f}원")
                print(f"   📊 유사도: {sim:.3f}")
                print(f"   📝 {event['content'][:60]}...")
                print()
                
        except Exception as e:
            print(f"❌ 오류: {e}")

def main():
    """메인 실행 함수"""
    print("🎵 이벤트 추천 시스템 - 모델 학습 데모")
    print("=" * 60)
    
    # 1. 데이터 로드
    df, meta_preprocessor = load_data()
    if df is None:
        return
    
    print(f"\n📊 데이터 정보:")
    print(f"• 이벤트 수: {len(df):,}개")
    print(f"• 컬럼 수: {len(df.columns)}개")
    print(f"• 주요 컬럼: {list(df.columns[:5])}")
    
    # 2. 텍스트 코퍼스 생성
    print(f"\n🔧 텍스트 전처리 중...")
    df['content_clean'] = df['content'].apply(preprocess_text)
    df['place_clean'] = df['place'].apply(preprocess_text)
    df['location_clean'] = df['loc_sigu'].apply(preprocess_text)
    
    text_corpus = (
        df['content_clean'].fillna('') + ' ' +
        df['place_clean'].fillna('') + ' ' +
        df['location_clean'].fillna('')
    )
    
    # 3. 모델 학습
    models = {}
    
    # TF-IDF 모델
    models['tfidf'] = build_tfidf_model(df, meta_preprocessor)
    
    # LSA 모델
    models['lsa'] = build_lsa_model(df, meta_preprocessor, text_corpus)
    
    print(f"\n✅ 총 {len(models)}개 모델 학습 완료!")
    
    # 4. 추천 테스트
    test_recommendations(models, df, meta_preprocessor)
    
    print("\n🎉 데모 완료!")
    print("💡 더 자세한 내용은 model_training_demo.ipynb 노트북을 확인하세요.")

if __name__ == "__main__":
    main() 