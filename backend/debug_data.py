#!/usr/bin/env python3

from user_preferences import preference_manager
from ranknet_model import ranknet_recommender

def debug_preference_data():
    print("=== 사용자 선호도 데이터 디버깅 ===")
    
    # 1. 전체 사용자 확인
    import sqlite3
    conn = sqlite3.connect(preference_manager.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT user_id, total_likes FROM users ORDER BY total_likes DESC")
    users = cursor.fetchall()
    print(f"\n📊 전체 사용자 목록:")
    for user_id, total_likes in users:
        print(f"  - {user_id}: {total_likes}개 좋아요")
    
    # 2. 좋아요 데이터 확인
    cursor.execute("SELECT COUNT(*) FROM user_likes WHERE is_liked = TRUE")
    total_likes = cursor.fetchone()[0]
    print(f"\n❤️ 전체 좋아요 수: {total_likes}개")
    
    # 3. 학습용 데이터 추출 테스트
    print(f"\n🧠 RankNet 학습용 데이터 추출 테스트:")
    for min_likes in [3, 5, 1]:
        print(f"\n--- min_likes={min_likes} ---")
        preference_data = preference_manager.get_preference_data_for_training(min_likes=min_likes)
        print(f"추출된 데이터: {len(preference_data)}개")
        
        if len(preference_data) > 0:
            print("데이터 샘플:")
            print(preference_data.head())
    
    # 4. RankNet 모델 상태 확인
    print(f"\n🤖 RankNet 모델 상태:")
    status = ranknet_recommender.get_model_status()
    for key, value in status.items():
        print(f"  - {key}: {value}")
    
    conn.close()

if __name__ == "__main__":
    debug_preference_data() 