"""
균등 가중치 추천 시스템 (Balanced Recommender)
내용 34% + 가격 33% + 위치 33% = 균형잡힌 추천
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import joblib

class BalancedBackendRecommender:
    def __init__(self, model_path):
        """균등 가중치 추천 시스템 초기화"""
        self.model_path = Path(model_path)
        self.models = None
        self.df = None
        self.meta_preprocessor = None
        
        # 정규화된 가중치 설정
        self.NORMALIZED_WEIGHTS = {
            'content': 0.34,   # 34%
            'price': 0.33,     # 33%
            'location': 0.33   # 33%
        }
        
        try:
            self.load_models()
            print("✅ Balanced recommender initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize balanced recommender: {e}")
            raise

    def load_models(self):
        """모델과 데이터 로드"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # 저장된 모델 로드
        saved_data = joblib.load(self.model_path)
        
        self.models = saved_data.get('models', {})
        self.df = saved_data.get('df')
        self.meta_preprocessor = saved_data.get('meta_preprocessor')
        
        # 필요한 컴포넌트들 추출
        self.tfidf_vectorizer = self.models.get('tfidf', {}).get('vectorizer')
        self.count_vectorizer = self.models.get('lsa', {}).get('count_vectorizer')
        self.svd = self.models.get('lsa', {}).get('svd')
        self.word2vec_hasher = self.models.get('word2vec', {}).get('hasher')
        
        # 특성 데이터 준비
        self._prepare_features()

    def _prepare_features(self):
        """추천에 필요한 특성 데이터 준비"""
        if self.df is None:
            raise ValueError("DataFrame not loaded")
        
        # 텍스트 특성들
        if 'tfidf' in self.models:
            self.X_text_tfidf = self.models['tfidf'].get('X_text')
        if 'lsa' in self.models:
            self.X_text_lsa = self.models['lsa'].get('X_reduced')
        if 'word2vec' in self.models:
            self.X_text_w2v = self.models['word2vec'].get('X_text')
        
        # 메타데이터 특성
        self.X_meta = self.models.get('meta_features')
        
        if self.X_meta is not None:
            # 가격과 위치 특성 분리
            self.price_features = self.X_meta[:, :1]  # 가격 특성만
            self.location_features = self.X_meta[:, 1:]  # 위치 특성들
        else:
            print("⚠️ Meta features not found, using dummy features")
            n_samples = len(self.df)
            self.price_features = np.random.rand(n_samples, 1)
            self.location_features = np.random.rand(n_samples, 10)

    def safe_toarray(self, matrix):
        """Sparse matrix든 dense array든 안전하게 array로 변환"""
        if hasattr(matrix, 'toarray'):
            return matrix.toarray()
        else:
            return matrix

    def encode_query_balanced(self, query):
        """쿼리를 균형있게 인코딩하는 함수"""
        keywords = query.get('keywords', '')
        price_max = query.get('price_max', 25000)
        location = query.get('location', 'unknown')
        
        if self.meta_preprocessor is None:
            print("⚠️ Meta preprocessor not found, using dummy encoding")
            # 더미 데이터 반환 (차원 맞춤)
            return keywords, np.array([price_max / 100000]).reshape(1, -1), np.array([0.5] * 10).reshape(1, -1)
        
        # 메타데이터 처리
        meta_df = pd.DataFrame([{
            'price_adv': price_max,
            'price_door': price_max,
            'loc_sigu': location
        }])
        
        try:
            meta_vec = self.meta_preprocessor.transform(meta_df)
            meta_array = self.safe_toarray(meta_vec)
            price_data = meta_array[0, :1]
            location_vec = meta_array[0, 1:]
            return keywords, price_data, location_vec
        except Exception as e:
            print(f"⚠️ Error in meta preprocessing: {e}, using dummy data")
            return keywords, np.array([price_max / 100000]).reshape(1, -1), np.array([0.5] * 10).reshape(1, -1)

    def recommend(self, query, model_type='tfidf', top_k=10):
        """균등 가중치 추천 실행"""
        try:
            keywords, price_data, location_vec = self.encode_query_balanced(query)
            
            # 모델별 특성 인코딩
            if model_type == 'tfidf' and hasattr(self, 'X_text_tfidf'):
                q_content = self.safe_toarray(self.tfidf_vectorizer.transform([keywords]))[0]
                content_sim = cosine_similarity([q_content], self.safe_toarray(self.X_text_tfidf))[0]
            elif model_type == 'lsa' and hasattr(self, 'X_text_lsa'):
                q_count = self.count_vectorizer.transform([keywords])
                q_lsa = self.svd.transform(q_count)[0]
                content_sim = cosine_similarity([q_lsa], self.X_text_lsa)[0]
            elif model_type == 'word2vec' and hasattr(self, 'X_text_w2v'):
                q_hash = self.safe_toarray(self.word2vec_hasher.transform([keywords]))[0]
                content_sim = cosine_similarity([q_hash], self.safe_toarray(self.X_text_w2v))[0]
            else:
                # 폴백: TF-IDF 사용 또는 더미 유사도
                if hasattr(self, 'X_text_tfidf'):
                    q_content = self.safe_toarray(self.tfidf_vectorizer.transform([keywords]))[0]
                    content_sim = cosine_similarity([q_content], self.safe_toarray(self.X_text_tfidf))[0]
                else:
                    print("⚠️ No text features available, using random similarity")
                    content_sim = np.random.rand(len(self.df))
            
            # 가격 및 위치 유사도 계산
            try:
                # 1차원으로 정확히 맞춤
                price_query = price_data.flatten() if hasattr(price_data, 'flatten') else price_data
                location_query = location_vec.flatten() if hasattr(location_vec, 'flatten') else location_vec
                
                price_sim = cosine_similarity([price_query], self.safe_toarray(self.price_features))[0]
                location_sim = cosine_similarity([location_query], self.safe_toarray(self.location_features))[0]
            except Exception as e:
                print(f"⚠️ Error in meta similarity calculation: {e}, using random similarity")
                price_sim = np.random.rand(len(self.df))
                location_sim = np.random.rand(len(self.df))
            
            # 정규화된 가중합 계산
            total_similarity = (
                self.NORMALIZED_WEIGHTS['content'] * content_sim +
                self.NORMALIZED_WEIGHTS['price'] * price_sim +
                self.NORMALIZED_WEIGHTS['location'] * location_sim
            )
            
            # 상위 k개 결과 선택
            top_indices = np.argsort(total_similarity)[::-1][:top_k]
            
            # 결과 DataFrame 구성
            results = []
            for idx in top_indices:
                event = self.df.iloc[idx]
                results.append({
                    'link': event.get('link', ''),
                    'content': event.get('content', ''),
                    'place': event.get('place', ''),
                    'date': event.get('date', ''),
                    'time': event.get('time', ''),
                    'price_adv': float(event.get('price_adv', 0)),
                    'price_door': float(event.get('price_door', 0)),
                    'loc_sigu': event.get('loc_sigu', ''),
                    'score': float(total_similarity[idx]),
                    'model_used': 'balanced',
                    'content_similarity': float(content_sim[idx]),
                    'price_similarity': float(price_sim[idx]),
                    'location_similarity': float(location_sim[idx]),
                    'weighted_content': float(self.NORMALIZED_WEIGHTS['content'] * content_sim[idx]),
                    'weighted_price': float(self.NORMALIZED_WEIGHTS['price'] * price_sim[idx]),
                    'weighted_location': float(self.NORMALIZED_WEIGHTS['location'] * location_sim[idx])
                })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            print(f"❌ Error in balanced recommendation: {e}")
            # 폴백: 랜덤 추천
            random_indices = np.random.choice(len(self.df), min(top_k, len(self.df)), replace=False)
            results = []
            for idx in random_indices:
                event = self.df.iloc[idx]
                results.append({
                    'link': event.get('link', ''),
                    'content': event.get('content', ''),
                    'place': event.get('place', ''),
                    'date': event.get('date', ''),
                    'time': event.get('time', ''),
                    'price_adv': float(event.get('price_adv', 0)),
                    'price_door': float(event.get('price_door', 0)),
                    'loc_sigu': event.get('loc_sigu', ''),
                    'score': 0.5,
                    'model_used': 'balanced_fallback'
                })
            return pd.DataFrame(results) 