#!/usr/bin/env python3

from user_preferences import preference_manager
from ranknet_model import ranknet_recommender

def train_ranknet_direct():
    print("=== RankNet 직접 학습 시도 ===")
    
    # 1. 학습 데이터 가져오기
    print("\n1. 학습 데이터 수집...")
    preference_data = preference_manager.get_preference_data_for_training(min_likes=3)
    print(f"수집된 데이터: {len(preference_data)}개")
    
    if len(preference_data) < 5:
        print("❌ 학습 데이터가 부족합니다.")
        return False
    
    # 2. 데이터 상세 확인
    print(f"\n2. 데이터 상세 분석...")
    print(f"컬럼: {list(preference_data.columns)}")
    print(f"is_liked 분포:")
    print(preference_data['is_liked'].value_counts())
    
    # 3. RankNet 학습 시도
    print(f"\n3. RankNet 모델 학습 시작...")
    try:
        success = ranknet_recommender.train(preference_data, epochs=20)
        if success:
            print("✅ RankNet 학습 성공!")
            
            # 4. 학습 후 상태 확인
            status = ranknet_recommender.get_model_status()
            print(f"\n4. 학습 후 모델 상태:")
            for key, value in status.items():
                print(f"  - {key}: {value}")
            
            return True
        else:
            print("❌ RankNet 학습 실패")
            return False
            
    except Exception as e:
        print(f"❌ RankNet 학습 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    train_ranknet_direct() 