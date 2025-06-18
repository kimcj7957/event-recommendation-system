#!/usr/bin/env python3
"""
새로운 scikit-learn 버전으로 모델을 다시 훈련하는 스크립트
데이터 품질 개선: 누락 데이터 및 중복 제거
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack
import joblib

def clean_data(df):
    """데이터 품질 개선: 누락 데이터 및 중복 제거"""
    print(f'원본 데이터 크기: {len(df)}')
    
    # 1. 본문(content)이 누락되거나 너무 짧은 데이터 제거
    df_clean = df.dropna(subset=['content']).copy()
    df_clean = df_clean[df_clean['content'].str.len() >= 10]  # 최소 10글자 이상
    print(f'본문 필터링 후: {len(df_clean)}')
    
    # 2. 동일한 본문을 가진 중복 데이터 제거 (첫 번째만 유지)
    df_clean = df_clean.drop_duplicates(subset=['content'], keep='first')
    print(f'중복 제거 후: {len(df_clean)}')
    
    # 3. 링크가 누락된 데이터 제거
    df_clean = df_clean.dropna(subset=['link'])
    df_clean = df_clean[df_clean['link'].str.len() > 0]
    print(f'링크 필터링 후: {len(df_clean)}')
    
    # 4. 장소 정보가 있는 데이터만 유지 (추천 품질 향상)
    df_clean = df_clean.dropna(subset=['place'])
    df_clean = df_clean[df_clean['place'].str.len() > 0]
    print(f'장소 필터링 후: {len(df_clean)}')
    
    return df_clean

def main():
    print('Loading data...')
    df = pd.read_csv('../merged_events_clean - merged_events_clean.csv')
    
    print('Cleaning data...')
    df = clean_data(df)
    
    print('Preprocessing...')
    # 날짜/시간 처리
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['time'] = pd.to_datetime(df['time'], format='%H:%M', errors='coerce').dt.time
    
    # 가격 처리
    df['price_adv'] = pd.to_numeric(df['in advance'], errors='coerce')
    df['price_door'] = pd.to_numeric(df['cover'], errors='coerce')
    
    mid_price = df[['price_adv', 'price_door']].stack().median()
    print(f'중간 가격: {mid_price}')
    
    df['price_adv'] = df['price_adv'].fillna(df['price_door'])
    df['price_door'] = df['price_door'].fillna(df['price_adv'])
    df[['price_adv','price_door']] = df[['price_adv','price_door']].fillna(mid_price)
    
    # 위치 처리
    loc = df['place'].fillna('').str.extract(r'^(?P<city>[^ ]+)\s*(?P<gu>[^ ]+)?')
    df['loc_sigu'] = loc['city'].fillna('') + ' ' + loc['gu'].fillna('')
    df.loc[df['loc_sigu'].str.strip()=='','loc_sigu'] = 'unknown'
    
    print('Creating text features...')
    text_corpus = (
        df['content'].fillna('') + ' ' +
        df['place'].fillna('')   + ' ' +
        df['loc_sigu'].fillna('')
    )
    
    tfidf = TfidfVectorizer(max_features=10_000,
                            ngram_range=(1,2),
                            min_df=2,  # 최소 출현 빈도 낮춤 (더 다양한 키워드)
                            stop_words='english')
    X_text = tfidf.fit_transform(text_corpus)
    print(f'TF-IDF 피처 수: {X_text.shape[1]}')
    
    print('Creating metadata features...')
    num_cols = ['price_adv', 'price_door']
    cat_cols = ['loc_sigu']
    
    pre = ColumnTransformer([
        ('num', MinMaxScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    
    X_meta = pre.fit_transform(df)
    X_all = hstack([X_text, X_meta]).tocsr()
    print(f'전체 피처 수: {X_all.shape[1]}')
    
    print('Training KNN model...')
    knn = NearestNeighbors(metric='cosine', n_neighbors=min(20, len(df)), n_jobs=-1)
    knn.fit(X_all)
    
    print('Saving model...')
    model_data = {
        'tfidf': tfidf,
        'pre': pre,
        'knn': knn,
        'df': df
    }
    
    # 새 모델 저장
    joblib.dump(model_data, '../model/recommender_ko_clean.joblib')
    print('Model saved to ../model/recommender_ko_clean.joblib')
    
    # 백업용으로 기존 모델 이름 변경 후 새 모델을 원래 이름으로 저장
    import shutil
    try:
        shutil.move('../model/recommender_ko.joblib', '../model/recommender_ko_backup.joblib')
        shutil.move('../model/recommender_ko_clean.joblib', '../model/recommender_ko.joblib')
        print('Model updated successfully!')
        print(f'최종 데이터 크기: {len(df)} 이벤트')
    except Exception as e:
        print(f'Warning: Could not update model file: {e}')

if __name__ == '__main__':
    main() 