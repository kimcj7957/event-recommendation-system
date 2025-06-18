import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path
import json
import random

class RankNet(nn.Module):
    """RankNet 모델 - 순위 학습을 위한 신경망"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super(RankNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # 출력층 - 점수 예측
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def predict_score(self, x):
        """점수 예측"""
        with torch.no_grad():
            return self.forward(x).squeeze()

class PersonalizedRankNetRecommender:
    """개인화된 RankNet 추천 시스템"""
    
    def __init__(self, model_path: str = "models/ranknet_model.pt"):
        self.model_path = Path(model_path)
        self.model = None
        self.feature_scaler = StandardScaler()
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words=None)
        self.location_encoder = LabelEncoder()
        self.is_trained = False
        
        # 특성 차원
        self.text_features_dim = 100
        self.numerical_features = ['price_adv', 'price_door', 'hour', 'day_of_week', 'month']
        self.categorical_features = ['location_encoded']
        self.total_features_dim = self.text_features_dim + len(self.numerical_features) + len(self.categorical_features)
        
        self.load_model()
    
    def extract_features(self, events_features: List[dict]) -> np.ndarray:
        """이벤트 특성으로부터 벡터 추출"""
        try:
            features_list = []
            
            for event_features in events_features:
                # 텍스트 특성 추출
                content = event_features.get('content', '')
                venue = event_features.get('venue', '')
                price = event_features.get('price', 0)
                
                # TF-IDF 벡터화
                text_features = self.vectorizer.transform([content + ' ' + venue]).toarray()[0]
                
                # 가격 특성 (정규화)
                price_feature = [min(price / 10000, 5.0)]  # 만원 단위로 정규화, 최대 5
                
                # 특성 결합
                combined_features = np.concatenate([text_features, price_feature])
                features_list.append(combined_features)
            
            features_array = np.array(features_list)
            
            # 학습 중이면 스케일러를 fit하고, 그렇지 않으면 transform만 수행
            if hasattr(self, '_is_training') and self._is_training:
                features_array = self.feature_scaler.fit_transform(features_array)
            else:
                # 스케일러가 fit되지 않았으면 그대로 반환
                if not hasattr(self.feature_scaler, 'scale_'):
                    print("[WARNING] StandardScaler not fitted, returning raw features")
                    return features_array
                features_array = self.feature_scaler.transform(features_array)
            
            return features_array
            
        except Exception as e:
            print(f"특성 추출 중 오류: {e}")
            # 기본 특성 반환
            return np.zeros((len(events_features), self.total_features_dim))
    
    def prepare_training_data(self, preference_data: pd.DataFrame):
        """학습용 데이터 준비 - 좋아요 데이터만 있는 경우 처리"""
        try:
            # 학습 모드 설정
            self._is_training = True
            
            users = preference_data['user_id'].unique()
            pairs_data = []
            
            print(f"사용자 수: {len(users)}")
            
            for user in users:
                user_data = preference_data[preference_data['user_id'] == user]
                liked_events = user_data[user_data['is_liked'] == True]
                disliked_events = user_data[user_data['is_liked'] == False]
                
                print(f"사용자 {user}: 좋아요 {len(liked_events)}개, 싫어요 {len(disliked_events)}개")
                
                # 좋아요만 있는 경우, 다른 사용자의 데이터를 negative로 사용
                if len(disliked_events) == 0:
                    print(f"사용자 {user}의 negative 샘플을 시간 순서로 생성")
                    # 시간 순서로 정렬
                    liked_events_sorted = liked_events.sort_values('liked_at')
                    events_list = liked_events_sorted['event_features_parsed'].tolist()
                    
                    # 시간 순서 기반으로 페어 생성 (나중에 좋아한 것이 더 선호된다고 가정)
                    for i in range(len(events_list)):
                        for j in range(i):
                            # i번째(나중) 이벤트가 j번째(이전) 이벤트보다 선호됨
                            pairs_data.append((events_list[i], events_list[j], 1))
                    
                    # 추가적으로 랜덤 페어도 생성 (같은 선호도로 간주)
                    if len(events_list) >= 2:
                        for _ in range(min(10, len(events_list) * 2)):  # 최대 10개의 랜덤 페어
                            i, j = random.sample(range(len(events_list)), 2)
                            pairs_data.append((events_list[i], events_list[j], 0.5))  # 중간 선호도
                else:
                    # 기존 로직: 좋아요 vs 싫어요
                    for liked_event in liked_events['event_features_parsed']:
                        for disliked_event in disliked_events['event_features_parsed']:
                            pairs_data.append((liked_event, disliked_event, 1))
            
            if len(pairs_data) == 0:
                print("학습용 페어 데이터가 생성되지 않았습니다.")
                return None, None
            
            print(f"생성된 학습 페어: {len(pairs_data)}개")
            
            # 특성 추출
            all_events = []
            for pair in pairs_data:
                all_events.extend([pair[0], pair[1]])
            
            # 중복 제거
            unique_events = []
            seen = set()
            for event in all_events:
                event_str = str(event)
                if event_str not in seen:
                    unique_events.append(event)
                    seen.add(event_str)
            
            print(f"고유 이벤트 수: {len(unique_events)}")
            
            # 텍스트 벡터라이저 학습
            texts = []
            for event in unique_events:
                content = event.get('content', '')
                venue = event.get('venue', '')
                texts.append(content + ' ' + venue)
            
            if len(texts) > 0:
                self.vectorizer.fit(texts)
                print(f"텍스트 벡터라이저 학습 완료: {len(texts)}개 텍스트")
            
            # 특성 추출
            event_features = self.extract_features(unique_events)
            
            # 이벤트-특성 매핑
            event_to_features = {str(event): features for event, features in zip(unique_events, event_features)}
            
            # 페어 데이터 구성
            X_pairs = []
            y_pairs = []
            
            for event1, event2, label in pairs_data:
                feat1 = event_to_features[str(event1)]
                feat2 = event_to_features[str(event2)]
                
                X_pairs.append(np.concatenate([feat1, feat2]))
                y_pairs.append(label)
            
            # 학습 모드 해제
            self._is_training = False
            
            return np.array(X_pairs), np.array(y_pairs)
            
        except Exception as e:
            print(f"학습 데이터 준비 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def train(self, preference_data: pd.DataFrame, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001):
        """RankNet 모델 학습"""
        X_pairs, y_pairs = self.prepare_training_data(preference_data)
        
        if len(X_pairs) == 0:
            print("충분한 학습 데이터가 없습니다.")
            return False
        
        # 모델 초기화
        pair_input_dim = X_pairs.shape[1]  # 두 개 이벤트의 특성을 연결한 차원
        self.model = RankNet(pair_input_dim)
        
        # 손실함수와 옵티마이저
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 데이터를 텐서로 변환
        X_tensor = torch.FloatTensor(X_pairs)
        y_tensor = torch.FloatTensor(y_pairs)
        
        # 학습
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            # 미니배치 학습
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                
                # 순전파
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                
                # 역전파
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / (len(X_tensor) // batch_size + 1)
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        self.is_trained = True
        self.save_model()
        print("RankNet 모델 학습 완료!")
        return True
    
    def predict_preferences(self, user_id: str, events_data: List[Dict]) -> List[float]:
        """사용자의 이벤트 선호도 점수 예측"""
        if not self.is_trained or self.model is None:
            # 모델이 학습되지 않은 경우 기본 점수 반환
            return [0.5] * len(events_data)
        
        # 특성 추출
        features = self.extract_features(events_data)
        
        if features.shape[0] == 0:
            return [0.5] * len(events_data)
        
        # 예측 (단일 이벤트 점수 예측을 위해 자기 자신과 페어 생성)
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for i in range(len(features)):
                # 각 이벤트를 자기 자신과 페어로 만들어 점수 예측
                # 실제로는 더 복잡한 방법이 필요하지만 간단한 구현
                feature_pair = np.concatenate([features[i], features[i]])
                feature_tensor = torch.FloatTensor(feature_pair).unsqueeze(0)
                
                score = torch.sigmoid(self.model(feature_tensor)).item()
                scores.append(score)
        
        return scores
    
    def save_model(self):
        """모델 저장"""
        if self.model is not None:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 모델과 전처리기 저장
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_architecture': {
                    'input_dim': self.total_features_dim * 2,  # 페어 입력
                    'hidden_dims': [128, 64, 32]
                },
                'feature_scaler': self.feature_scaler,
                'vectorizer': self.vectorizer,
                'location_encoder': self.location_encoder,
                'is_trained': self.is_trained
            }, self.model_path)
    
    def load_model(self):
        """모델 로드"""
        if self.model_path.exists():
            try:
                checkpoint = torch.load(self.model_path, map_location='cpu')
                
                # 모델 아키텍처 복원
                arch = checkpoint['model_architecture']
                self.model = RankNet(arch['input_dim'], arch['hidden_dims'])
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # 전처리기 복원
                self.feature_scaler = checkpoint['feature_scaler']
                self.vectorizer = checkpoint['vectorizer']
                self.location_encoder = checkpoint['location_encoder']
                self.is_trained = checkpoint['is_trained']
                
                print("RankNet 모델이 성공적으로 로드되었습니다.")
                
            except Exception as e:
                print(f"모델 로드 실패: {e}")
                self.is_trained = False
    
    def get_model_status(self) -> Dict:
        """모델 상태 정보 반환"""
        return {
            'is_trained': self.is_trained,
            'model_exists': self.model is not None,
            'model_path': str(self.model_path),
            'total_features_dim': self.total_features_dim
        }

# 전역 RankNet 추천 시스템 인스턴스
ranknet_recommender = PersonalizedRankNetRecommender() 