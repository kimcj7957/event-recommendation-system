#!/usr/bin/env python3
"""
정규화된 가중합 추천 시스템 데모

가중치 설정:
- 내용: 34%
- 가격: 33%  
- 위치: 33%
- 총합: 100% (1.0)

최대 유사도가 1일 때: 0.34 × 1 + 0.33 × 1 + 0.33 × 1 = 1.0
"""

import pandas as pd
import numpy as np

def demonstrate_normalized_weights():
    """정규화된 가중합 시연"""
    print("🎯 정규화된 가중합 추천 시스템")
    print("=" * 50)
    
    # 가중치 설정
    weights = {
        'content': 0.34,
        'price': 0.33,
        'location': 0.33
    }
    
    print(f"\n📊 가중치 설정:")
    print(f"   내용(Content): {weights['content']:.2f} (34%)")
    print(f"   가격(Price): {weights['price']:.2f} (33%)")
    print(f"   위치(Location): {weights['location']:.2f} (33%)")
    print(f"   ✅ 총합: {sum(weights.values()):.2f} (100%)")
    
    # 시나리오 테스트
    print(f"\n🧪 시나리오 테스트:")
    
    scenarios = [
        {
            'name': '완벽한 매칭',
            'content_sim': 1.0,
            'price_sim': 1.0,
            'location_sim': 1.0
        },
        {
            'name': '부분적 매칭',
            'content_sim': 0.8,
            'price_sim': 0.9,
            'location_sim': 0.7
        },
        {
            'name': '내용 중심 매칭',
            'content_sim': 1.0,
            'price_sim': 0.5,
            'location_sim': 0.3
        },
        {
            'name': '가격 중심 매칭',
            'content_sim': 0.4,
            'price_sim': 1.0,
            'location_sim': 0.6
        }
    ]
    
    for scenario in scenarios:
        name = scenario['name']
        content_sim = scenario['content_sim']
        price_sim = scenario['price_sim']
        location_sim = scenario['location_sim']
        
        # 가중합 계산
        weighted_content = weights['content'] * content_sim
        weighted_price = weights['price'] * price_sim
        weighted_location = weights['location'] * location_sim
        total_similarity = weighted_content + weighted_price + weighted_location
        
        print(f"\n📋 {name}:")
        print(f"   내용 유사도: {content_sim:.1f} → 가중: {weighted_content:.3f}")
        print(f"   가격 유사도: {price_sim:.1f} → 가중: {weighted_price:.3f}")
        print(f"   위치 유사도: {location_sim:.1f} → 가중: {weighted_location:.3f}")
        print(f"   🎯 총 유사도: {total_similarity:.3f}")
    
    print(f"\n💡 핵심 포인트:")
    print(f"   • 모든 유사도가 1.0일 때 총 유사도는 정확히 1.0")
    print(f"   • 각 특성의 기여도가 명확히 정의됨")
    print(f"   • 사용자는 각 요소의 중요도를 정확히 이해 가능")
    print(f"   • 백엔드와 프론트엔드에서 일관된 결과 보장")

if __name__ == "__main__":
    demonstrate_normalized_weights() 