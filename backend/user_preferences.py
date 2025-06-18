import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

class UserPreferenceManager:
    """사용자 선호도 관리 시스템"""
    
    def __init__(self, db_path: str = "user_preferences.db"):
        self.db_path = Path(db_path)
        self.init_database()
    
    def init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 사용자 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_likes INTEGER DEFAULT 0
            )
        ''')
        
        # 좋아요 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_likes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                event_link TEXT,
                event_content TEXT,
                event_features TEXT,  -- JSON으로 저장된 이벤트 특성
                liked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_liked BOOLEAN DEFAULT TRUE,
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                UNIQUE(user_id, event_link)
            )
        ''')
        
        # 검색 히스토리 테이블 (추천 개선용)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                search_query TEXT,  -- JSON으로 저장된 검색 쿼리
                clicked_events TEXT,  -- 클릭한 이벤트들
                searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_or_create_user(self, user_id: str) -> bool:
        """사용자 생성 또는 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 사용자 존재 확인
        cursor.execute('SELECT user_id FROM users WHERE user_id = ?', (user_id,))
        exists = cursor.fetchone() is not None
        
        if not exists:
            cursor.execute('INSERT INTO users (user_id) VALUES (?)', (user_id,))
            conn.commit()
        
        conn.close()
        return not exists  # 새로 생성된 경우 True
    
    def toggle_like(self, user_id: str, event_link: str, event_data: Dict) -> Dict:
        """좋아요 토글 (추가/제거)"""
        self.get_or_create_user(user_id)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 현재 좋아요 상태 확인
        cursor.execute('''
            SELECT is_liked FROM user_likes 
            WHERE user_id = ? AND event_link = ?
        ''', (user_id, event_link))
        
        result = cursor.fetchone()
        
        # 이벤트 특성 JSON으로 저장
        event_features = {
            'content': event_data.get('content', ''),
            'place': event_data.get('place', ''),
            'price_adv': event_data.get('price_adv', 0),
            'price_door': event_data.get('price_door', 0),
            'date': event_data.get('date', ''),
            'time': event_data.get('time', '')
        }
        
        if result is None:
            # 새로운 좋아요 추가
            cursor.execute('''
                INSERT INTO user_likes (user_id, event_link, event_content, event_features, is_liked)
                VALUES (?, ?, ?, ?, TRUE)
            ''', (user_id, event_link, event_data.get('content', ''), json.dumps(event_features)))
            new_status = True
        else:
            # 좋아요 상태 토글
            new_status = not result[0]
            cursor.execute('''
                UPDATE user_likes 
                SET is_liked = ?, liked_at = CURRENT_TIMESTAMP
                WHERE user_id = ? AND event_link = ?
            ''', (new_status, user_id, event_link))
        
        # 사용자 총 좋아요 수 업데이트
        cursor.execute('''
            UPDATE users SET total_likes = (
                SELECT COUNT(*) FROM user_likes 
                WHERE user_id = ? AND is_liked = TRUE
            ) WHERE user_id = ?
        ''', (user_id, user_id))
        
        conn.commit()
        conn.close()
        
        return {
            'user_id': user_id,
            'event_link': event_link,
            'is_liked': new_status,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_user_likes(self, user_id: str) -> List[Dict]:
        """사용자의 좋아요 목록 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT event_link, event_content, event_features, liked_at, is_liked
            FROM user_likes 
            WHERE user_id = ? AND is_liked = TRUE
            ORDER BY liked_at DESC
        ''', (user_id,))
        
        likes = []
        for row in cursor.fetchall():
            likes.append({
                'event_link': row[0],
                'event_content': row[1],
                'event_features': json.loads(row[2]),
                'liked_at': row[3],
                'is_liked': row[4]
            })
        
        conn.close()
        return likes
    
    def get_user_stats(self, user_id: str) -> Dict:
        """사용자 통계 정보"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 기본 통계
        cursor.execute('SELECT total_likes, created_at FROM users WHERE user_id = ?', (user_id,))
        user_info = cursor.fetchone()
        
        if not user_info:
            conn.close()
            return {'exists': False}
        
        # 좋아요 트렌드 (최근 30일)
        cursor.execute('''
            SELECT DATE(liked_at) as date, COUNT(*) as count
            FROM user_likes 
            WHERE user_id = ? AND is_liked = TRUE 
            AND datetime(liked_at) >= datetime('now', '-30 days')
            GROUP BY DATE(liked_at)
            ORDER BY date DESC
        ''', (user_id,))
        
        daily_likes = cursor.fetchall()
        
        conn.close()
        
        return {
            'exists': True,
            'total_likes': user_info[0],
            'created_at': user_info[1],
            'daily_likes': [{'date': row[0], 'count': row[1]} for row in daily_likes]
        }
    
    def get_preference_data_for_training(self, min_likes: int = 5) -> pd.DataFrame:
        """RankNet 학습용 선호도 데이터 추출"""
        conn = sqlite3.connect(self.db_path)
        
        # 충분한 좋아요를 가진 사용자들의 데이터만 추출
        query = '''
            SELECT u.user_id, ul.event_features, ul.is_liked, ul.liked_at
            FROM users u
            JOIN user_likes ul ON u.user_id = ul.user_id
            WHERE u.total_likes >= ?
            ORDER BY u.user_id, ul.liked_at
        '''
        
        print(f"[DEBUG] 학습 데이터 쿼리 실행: min_likes={min_likes}")
        df = pd.read_sql_query(query, conn, params=(min_likes,))
        conn.close()
        
        print(f"[DEBUG] 쿼리 결과: {len(df)}개 행")
        if len(df) > 0:
            print(f"[DEBUG] 사용자별 데이터 분포:")
            user_counts = df['user_id'].value_counts()
            for user_id, count in user_counts.items():
                print(f"  - {user_id}: {count}개")
        
        if len(df) == 0:
            print("[DEBUG] 빈 DataFrame 반환")
            return pd.DataFrame()
        
        # JSON 특성 데이터 파싱
        try:
            df['event_features_parsed'] = df['event_features'].apply(json.loads)
            print(f"[DEBUG] JSON 파싱 성공")
        except Exception as e:
            print(f"[DEBUG] JSON 파싱 실패: {e}")
        
        return df
    
    def save_search_history(self, user_id: str, search_query: Dict, clicked_events: List[str]):
        """검색 히스토리 저장"""
        self.get_or_create_user(user_id)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO search_history (user_id, search_query, clicked_events)
            VALUES (?, ?, ?)
        ''', (user_id, json.dumps(search_query), json.dumps(clicked_events)))
        
        conn.commit()
        conn.close()

# 전역 인스턴스
preference_manager = UserPreferenceManager() 