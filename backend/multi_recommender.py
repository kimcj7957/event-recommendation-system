import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import joblib
import warnings
warnings.filterwarnings("ignore")

# Word2Vec import (optional, will fall back if not available)
try:
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess
    WORD2VEC_AVAILABLE = True
except ImportError:
    WORD2VEC_AVAILABLE = False
    print("Warning: gensim not available, using Word2Vec alternative")

# Simple Word2Vec alternative using scikit-learn
from sklearn.feature_extraction.text import HashingVectorizer
import re

def simple_preprocess_alt(text):
    """Simple text preprocessing without gensim"""
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Split into words and remove empty strings
    return [word for word in text.split() if len(word) > 1]

class MultiModelRecommender:
    """다중 모델 지원 추천 시스템"""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.models = {}
        self.df = None
        self.meta_preprocessor = None
        self.mid_price = 25000.0
        
    def load_base_data(self):
        """기본 데이터와 전처리기 로드"""
        if self.df is None:
            base_model = joblib.load(self.model_path)
            self.df = base_model['df']
            self.meta_preprocessor = base_model['pre']
        
    def build_tfidf_model(self):
        """TF-IDF 기반 모델 구축"""
        print("Building TF-IDF model...")
        self.load_base_data()
        
        # 텍스트 코퍼스 생성
        text_corpus = (
            self.df['content'].fillna('') + ' ' +
            self.df['place'].fillna('') + ' ' +
            self.df['loc_sigu'].fillna('')
        )
        
        # TF-IDF 벡터화
        tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            stop_words='english'
        )
        X_text = tfidf.fit_transform(text_corpus)
        
        # 메타데이터와 결합
        X_meta = self.meta_preprocessor.transform(self.df)
        X_combined = hstack([X_text, X_meta]).tocsr()
        
        # KNN 모델 학습
        knn = NearestNeighbors(metric='cosine', n_neighbors=20, n_jobs=-1)
        knn.fit(X_combined)
        
        self.models['tfidf'] = {
            'vectorizer': tfidf,
            'knn': knn,
            'type': 'knn'
        }
        
    def build_lsa_model(self):
        """LSA (SVD) 기반 모델 구축"""
        print("Building LSA model...")
        self.load_base_data()
        
        # 텍스트 코퍼스 생성
        text_corpus = (
            self.df['content'].fillna('') + ' ' +
            self.df['place'].fillna('') + ' ' +
            self.df['loc_sigu'].fillna('')
        )
        
        # Count 벡터화 (LSA는 보통 TF-IDF보다 Count에서 더 잘 작동)
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
        X_meta = self.meta_preprocessor.transform(self.df)
        # X_meta가 sparse matrix인지 dense array인지 확인
        if hasattr(X_meta, 'toarray'):
            X_meta = X_meta.toarray()
        X_combined = np.hstack([X_text_reduced, X_meta])
        
        # KNN 모델 학습
        knn = NearestNeighbors(metric='cosine', n_neighbors=20, n_jobs=-1)
        knn.fit(X_combined)
        
        self.models['lsa'] = {
            'count_vectorizer': count_vec,
            'svd': svd,
            'knn': knn,
            'type': 'knn'
        }
        
    def build_word2vec_model(self):
        """Word2Vec 대안 모델 구축 (HashingVectorizer 사용)"""
        print("Building Word2Vec alternative model...")
        self.load_base_data()
        
        # 텍스트 코퍼스 생성
        text_corpus = (
            self.df['content'].fillna('') + ' ' +
            self.df['place'].fillna('') + ' ' +
            self.df['loc_sigu'].fillna('')
        )
        
        # HashingVectorizer로 텍스트 임베딩 (Word2Vec 대안)
        hasher = HashingVectorizer(
            n_features=1000,
            ngram_range=(1, 3),
            binary=False,
            norm='l2',
            lowercase=True,
            stop_words='english'
        )
        X_text = hasher.fit_transform(text_corpus)
        
        # 메타데이터와 결합
        X_meta = self.meta_preprocessor.transform(self.df)
        X_combined = hstack([X_text, X_meta]).tocsr()
        
        # KNN 모델 학습
        knn = NearestNeighbors(metric='cosine', n_neighbors=20, n_jobs=-1)
        knn.fit(X_combined)
        
        self.models['word2vec'] = {
            'hasher': hasher,
            'knn': knn,
            'type': 'knn'
        }
        
    def build_hybrid_model(self):
        """하이브리드 모델 (여러 모델 조합)"""
        print("Building Hybrid model...")
        
        # 개별 모델들이 구축되어 있는지 확인
        required_models = ['tfidf', 'lsa', 'word2vec']
        available_models = [model for model in required_models if model in self.models]
        
        if len(available_models) < 2:
            print("Not enough models available for hybrid approach")
            return
            
        self.models['hybrid'] = {
            'component_models': available_models,
            'type': 'hybrid'
        }
        
    def encode_query(self, query: dict, model_name: str):
        """쿼리를 모델별로 인코딩"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
            
        model = self.models[model_name]
        
        # 기본 값 설정
        keywords = query.get('keywords', '').strip()
        price_max = query.get('price_max', self.mid_price)
        location = query.get('location', 'unknown').strip()
        
        if not keywords:
            keywords = ''
        if not location:
            location = 'unknown'
            
        # 메타데이터 벡터 생성
        meta_df = pd.DataFrame([{
            'price_adv': price_max,
            'price_door': price_max,
            'loc_sigu': location
        }])
        meta_vec = self.meta_preprocessor.transform(meta_df)
        
        # 모델별 텍스트 인코딩
        if model_name == 'tfidf':
            text_vec = model['vectorizer'].transform([keywords])
            return hstack([text_vec, meta_vec])
            
        elif model_name == 'lsa':
            count_vec = model['count_vectorizer'].transform([keywords])
            text_reduced = model['svd'].transform(count_vec)
            meta_vec_processed = meta_vec
            # meta_vec가 sparse matrix인지 dense array인지 확인
            if hasattr(meta_vec_processed, 'toarray'):
                meta_vec_processed = meta_vec_processed.toarray()
            return np.hstack([text_reduced, meta_vec_processed])
            
        elif model_name == 'word2vec':
            # Word2Vec 대안 (HashingVectorizer 사용)
            text_vec = model['hasher'].transform([keywords])
            return hstack([text_vec, meta_vec])
            
        else:
            raise ValueError(f"Unsupported model for encoding: {model_name}")
    
    def recommend(self, query: dict, model_name: str = 'tfidf', top_k: int = 5):
        """모델별 추천 실행"""
        try:
            self.load_base_data()
            
            if model_name == 'hybrid':
                return self._recommend_hybrid(query, top_k)
            
            if model_name not in self.models:
                # 요청된 모델이 없으면 기본값으로 대체
                print(f"Model {model_name} not found, using tfidf")
                model_name = 'tfidf'
                
            # 쿼리 벡터 생성
            q_vec = self.encode_query(query, model_name)
            
            # KNN 검색
            model = self.models[model_name]
            distances, indices = model['knn'].kneighbors(q_vec, n_neighbors=top_k)
            
            # 결과 생성
            recs = self.df.iloc[indices[0]].copy()
            recs['score'] = 1 - distances[0]  # 코사인 유사도
            recs['model_used'] = model_name
            
            # 안전한 데이터 처리
            recs['date'] = recs['date'].astype(str).replace('NaT', 'unknown')
            recs['time'] = recs['time'].astype(str).replace('NaT', 'unknown')
            
            result_cols = ['link', 'content', 'place', 'date', 'time', 'price_adv', 'price_door', 'score', 'model_used']
            result = recs[result_cols].fillna({
                'link': '', 'content': '', 'place': '',
                'date': 'unknown', 'time': 'unknown',
                'price_adv': 0, 'price_door': 0, 'score': 0, 'model_used': model_name
            })
            
            return result
            
        except Exception as e:
            print(f"Error in recommend function: {e}")
            # 에러 시 빈 결과 반환
            return pd.DataFrame(columns=['link', 'content', 'place', 'date', 'time', 'price_adv', 'price_door', 'score', 'model_used'])
    
    def _recommend_hybrid(self, query: dict, top_k: int = 5):
        """하이브리드 추천 (여러 모델 결과 조합)"""
        if 'hybrid' not in self.models:
            return self.recommend(query, 'tfidf', top_k)
            
        component_models = self.models['hybrid']['component_models']
        all_recommendations = []
        
        # 각 모델에서 추천 결과 수집
        for model_name in component_models:
            try:
                recs = self.recommend(query, model_name, top_k * 2)  # 더 많이 가져와서 조합
                recs['model_source'] = model_name
                all_recommendations.append(recs)
            except Exception as e:
                print(f"Error in {model_name} model: {e}")
                continue
        
        if not all_recommendations:
            return self.recommend(query, 'tfidf', top_k)
        
        # 결과 조합 및 재순위
        combined = pd.concat(all_recommendations, ignore_index=True)
        
        # 중복 제거 (같은 링크) 및 점수 조합
        unique_results = []
        seen_links = set()
        
        for _, row in combined.iterrows():
            if row['link'] not in seen_links:
                seen_links.add(row['link'])
                row['model_used'] = 'hybrid'
                unique_results.append(row)
        
        # 점수 기준 정렬 후 상위 k개 반환
        result_df = pd.DataFrame(unique_results)
        if len(result_df) > 0:
            result_df = result_df.sort_values('score', ascending=False).head(top_k)
        
        return result_df
    
    def get_available_models(self):
        """사용 가능한 모델 목록 반환"""
        return list(self.models.keys())
    
    def initialize_all_models(self):
        """모든 모델 초기화"""
        print("Initializing all recommendation models...")
        
        self.build_tfidf_model()
        self.build_lsa_model()
        
        # Word2Vec 대안 항상 빌드 (gensim 없어도 동작)
        self.build_word2vec_model()
            
        self.build_hybrid_model()
        
        print(f"Available models: {self.get_available_models()}")

# 전역 추천 시스템 인스턴스
MODEL_PATH = Path(__file__).parent.parent / 'model' / 'recommender_ko.joblib'
multi_recommender = MultiModelRecommender(MODEL_PATH)

def get_recommender():
    """전역 추천 시스템 인스턴스 반환"""
    return multi_recommender

def initialize_models():
    """모델 초기화 함수"""
    multi_recommender.initialize_all_models() 