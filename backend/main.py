from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from pathlib import Path
import uuid

# 기존 단일 모델과 새로운 다중 모델 시스템 임포트
try:
    from recommender import recommend as single_recommend
    SINGLE_MODEL_AVAILABLE = True
except ImportError:
    SINGLE_MODEL_AVAILABLE = False

try:
    from multi_recommender import get_recommender, initialize_models
    MULTI_MODEL_AVAILABLE = True
except ImportError:
    MULTI_MODEL_AVAILABLE = False

# 사용자 선호도 및 RankNet 시스템 임포트
try:
    from user_preferences import preference_manager
    from ranknet_model import ranknet_recommender
    PERSONALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Personalization system not available: {e}")
    PERSONALIZATION_AVAILABLE = False

app = FastAPI(
    title="Multi-Model Event Recommendation API with Personalization",
    description="이벤트 추천 시스템 - 다중 모델 지원 + 개인화 RankNet (TF-IDF, LSA, Word2Vec, Hybrid, RankNet)",
    version="3.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://localhost:3002",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendationQuery(BaseModel):
    keywords: str = ""
    price_max: Optional[float] = None
    location: str = ""
    model: str = "tfidf"  # 모델 선택 추가
    top_k: int = 5
    user_id: Optional[str] = None  # 개인화를 위한 사용자 ID

class LikeRequest(BaseModel):
    user_id: str
    event_link: str
    event_data: dict  # 이벤트 정보

class RecommendationResponse(BaseModel):
    query: dict
    recommendations: List[dict]
    total_count: int
    model_used: str
    available_models: List[str]
    personalized: bool = False  # 개인화 추천 여부
    user_stats: Optional[dict] = None

# 다중 모델 시스템 초기화
multi_recommender = None
if MULTI_MODEL_AVAILABLE:
    try:
        print("Initializing multi-model recommendation system...")
        initialize_models()
        multi_recommender = get_recommender()
        print("Multi-model system initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize multi-model system: {e}")
        MULTI_MODEL_AVAILABLE = False

@app.on_event("startup")
async def startup_event():
    print("🚀 Multi-Model Event Recommendation API Starting Up...")
    if MULTI_MODEL_AVAILABLE:
        available_models = multi_recommender.get_available_models()
        if PERSONALIZATION_AVAILABLE and ranknet_recommender.is_trained:
            available_models.append("ranknet")
        print(f"✅ Available models: {available_models}")
    elif SINGLE_MODEL_AVAILABLE:
        print("⚠️  Running with single TF-IDF model only")
    else:
        print("❌ No recommendation models available")
    
    if PERSONALIZATION_AVAILABLE:
        print("✅ Personalization system available")
    else:
        print("⚠️  Personalization system not available")

@app.get("/")
async def root():
    return {
        "message": "Multi-Model Event Recommendation API with Personalization",
        "version": "3.0.0",
        "multi_model_available": MULTI_MODEL_AVAILABLE,
        "single_model_available": SINGLE_MODEL_AVAILABLE,
        "personalization_available": PERSONALIZATION_AVAILABLE
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "API is working properly",
        "multi_model_system": MULTI_MODEL_AVAILABLE,
        "single_model_fallback": SINGLE_MODEL_AVAILABLE,
        "personalization_system": PERSONALIZATION_AVAILABLE
    }

@app.get("/models")
async def get_available_models():
    """사용 가능한 추천 모델 목록 반환"""
    if MULTI_MODEL_AVAILABLE and multi_recommender:
        available_models = multi_recommender.get_available_models()
    else:
        available_models = ["tfidf"] if SINGLE_MODEL_AVAILABLE else []
    
    # RankNet 모델 추가 (학습된 경우)
    if PERSONALIZATION_AVAILABLE and ranknet_recommender.is_trained:
        available_models.append("ranknet")
    
    model_descriptions = {
        "tfidf": "TF-IDF: 키워드 빈도 기반 추천 (빠름, 정확한 키워드 매칭)",
        "lsa": "LSA: 잠재 의미 분석 기반 추천 (의미적 유사성 고려)",
        "word2vec": "Word2Vec: 단어 임베딩 기반 추천 (단어 간 의미 관계 고려)",
        "hybrid": "Hybrid: 여러 모델 조합 추천 (종합적 결과)",
        "ranknet": "RankNet: 개인화 딥러닝 추천 (사용자 선호도 학습 기반)"
    }
    
    return {
        "available_models": available_models,
        "model_descriptions": {k: v for k, v in model_descriptions.items() if k in available_models}
    }

# 사용자 선호도 관련 API
@app.post("/like")
async def toggle_like(request: LikeRequest):
    """좋아요 토글 API"""
    if not PERSONALIZATION_AVAILABLE:
        raise HTTPException(status_code=501, detail="Personalization system not available")
    
    try:
        result = preference_manager.toggle_like(
            request.user_id, 
            request.event_link, 
            request.event_data
        )
        
        # 충분한 데이터가 수집된 경우 RankNet 모델 재학습 체크
        user_stats = preference_manager.get_user_stats(request.user_id)
        if user_stats['total_likes'] >= 3:  # 3개 이상 좋아요 시 학습 고려 (5개에서 3개로 완화)
            await check_and_retrain_ranknet()
        
        return {
            "success": True,
            "result": result,
            "user_stats": user_stats
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to toggle like: {str(e)}")

@app.get("/user/{user_id}/likes")
async def get_user_likes(user_id: str):
    """사용자 좋아요 목록 조회"""
    if not PERSONALIZATION_AVAILABLE:
        raise HTTPException(status_code=501, detail="Personalization system not available")
    
    try:
        likes = preference_manager.get_user_likes(user_id)
        stats = preference_manager.get_user_stats(user_id)
        
        return {
            "user_id": user_id,
            "likes": likes,
            "stats": stats
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user likes: {str(e)}")

@app.get("/user/{user_id}/stats")
async def get_user_stats(user_id: str):
    """사용자 통계 정보 조회"""
    if not PERSONALIZATION_AVAILABLE:
        raise HTTPException(status_code=501, detail="Personalization system not available")
    
    try:
        stats = preference_manager.get_user_stats(user_id)
        
        # RankNet 모델 상태 추가
        if ranknet_recommender.is_trained:
            stats['ranknet_available'] = True
        else:
            stats['ranknet_available'] = False
            
        return stats
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user stats: {str(e)}")

async def check_and_retrain_ranknet():
    """RankNet 모델 재학습 체크 및 실행"""
    if not PERSONALIZATION_AVAILABLE:
        return
    
    try:
        # 충분한 학습 데이터가 있는지 확인 (조건 완화)
        preference_data = preference_manager.get_preference_data_for_training(min_likes=3)  # 3개로 완화
        
        print(f"RankNet 학습 데이터 확인: {len(preference_data)}개 데이터")
        
        if len(preference_data) >= 5:  # 최소 5개 선호도 데이터로 완화
            print("충분한 데이터 확보, RankNet 모델 재학습 시작...")
            success = ranknet_recommender.train(preference_data, epochs=30)  # 에포크도 줄임
            if success:
                print("RankNet 모델 재학습 완료!")
            else:
                print("RankNet 모델 재학습 실패")
        else:
            print(f"RankNet 학습용 데이터 부족: {len(preference_data)}개 (최소 5개 필요)")
    
    except Exception as e:
        print(f"RankNet 재학습 중 오류: {e}")
        import traceback
        traceback.print_exc()

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_events(query: RecommendationQuery):
    """이벤트 추천 API - 다중 모델 + 개인화 지원"""
    try:
        query_dict = {
            "keywords": query.keywords,
            "price_max": query.price_max,
            "location": query.location
        }
        
        # 사용자 ID가 없으면 새로 생성
        user_id = query.user_id or str(uuid.uuid4())
        
        # 개인화 추천 (RankNet) 사용 조건 체크
        use_personalized = (
            PERSONALIZATION_AVAILABLE and 
            query.model == "ranknet" and 
            ranknet_recommender.is_trained and
            query.user_id is not None
        )
        
        if use_personalized:
            # RankNet 기반 개인화 추천
            try:
                # 먼저 기본 추천을 얻고
                base_recommendations_df = multi_recommender.recommend(
                    query_dict, 
                    model_name="hybrid",  # 기본으로 hybrid 사용
                    top_k=query.top_k * 2  # 더 많은 후보 생성
                )
                
                # 개인화 점수 적용
                base_recommendations = base_recommendations_df.to_dict(orient='records')
                personalized_scores = ranknet_recommender.predict_preferences(
                    user_id, base_recommendations
                )
                
                # 개인화 점수로 재정렬
                for i, rec in enumerate(base_recommendations):
                    rec['personalized_score'] = personalized_scores[i]
                    rec['original_score'] = rec.get('score', 0.5)
                    # 개인화 점수와 원래 점수를 조합
                    rec['score'] = 0.7 * personalized_scores[i] + 0.3 * rec.get('score', 0.5)
                    rec['model_used'] = 'ranknet'
                
                # 개인화 점수로 정렬하고 상위 k개 선택
                recommendations = sorted(
                    base_recommendations, 
                    key=lambda x: x['score'], 
                    reverse=True
                )[:query.top_k]
                
                model_used = "ranknet"
                personalized = True
                
            except Exception as e:
                print(f"RankNet 추천 실패, 기본 모델로 폴백: {e}")
                use_personalized = False
        
        if not use_personalized:
            # 기본 다중 모델 시스템 사용
            if MULTI_MODEL_AVAILABLE and multi_recommender:
                available_models = multi_recommender.get_available_models()
                
                # 요청된 모델이 사용 가능한지 확인
                actual_model = query.model if query.model in available_models else available_models[0]
                
                # 추천 실행
                recommendations_df = multi_recommender.recommend(
                    query_dict, 
                    model_name=actual_model, 
                    top_k=query.top_k
                )
                
                recommendations = recommendations_df.to_dict(orient='records')
                model_used = actual_model
                personalized = False
                
            # 단일 모델 폴백
            elif SINGLE_MODEL_AVAILABLE:
                recommendations_df = single_recommend(query_dict, top_k=query.top_k)
                recommendations = recommendations_df.to_dict(orient='records')
                for rec in recommendations:
                    rec['model_used'] = 'tfidf'
                model_used = "tfidf"
                personalized = False
                
            else:
                raise HTTPException(status_code=500, detail="No recommendation models available")
        
        # 사용자 통계 정보 추가
        user_stats = None
        if PERSONALIZATION_AVAILABLE and query.user_id:
            user_stats = preference_manager.get_user_stats(query.user_id)
        
        # 응답 생성
        available_models = []
        if MULTI_MODEL_AVAILABLE:
            available_models = multi_recommender.get_available_models()
        elif SINGLE_MODEL_AVAILABLE:
            available_models = ["tfidf"]
        
        if PERSONALIZATION_AVAILABLE and ranknet_recommender.is_trained:
            available_models.append("ranknet")
        
        return RecommendationResponse(
            query=query_dict,
            recommendations=recommendations,
            total_count=len(recommendations),
            model_used=model_used,
            available_models=available_models,
            personalized=personalized,
            user_stats=user_stats
        )
        
    except Exception as e:
        print(f"Error in recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.post("/recommend/compare")
async def compare_models(query: RecommendationQuery):
    """여러 모델의 추천 결과 비교"""
    if not MULTI_MODEL_AVAILABLE or not multi_recommender:
        raise HTTPException(status_code=400, detail="Multi-model system not available")
    
    try:
        query_dict = {
            "keywords": query.keywords,
            "price_max": query.price_max,
            "location": query.location
        }
        
        available_models = multi_recommender.get_available_models()
        comparison_results = {}
        
        # 각 모델별로 추천 실행
        for model_name in available_models:
            if model_name != 'hybrid':  # hybrid는 다른 모델들의 조합이므로 제외
                try:
                    recommendations_df = multi_recommender.recommend(
                        query_dict, 
                        model_name=model_name, 
                        top_k=query.top_k
                    )
                    comparison_results[model_name] = recommendations_df.to_dict(orient='records')
                except Exception as e:
                    print(f"Error with model {model_name}: {e}")
                    comparison_results[model_name] = {"error": str(e)}
        
        return {
            "query": query_dict,
            "model_comparisons": comparison_results,
            "available_models": available_models
        }
        
    except Exception as e:
        print(f"Error in model comparison: {e}")
        raise HTTPException(status_code=500, detail=f"Model comparison failed: {str(e)}")

@app.post("/train-ranknet")
async def manual_train_ranknet():
    """수동 RankNet 학습 트리거 (디버깅용)"""
    if not PERSONALIZATION_AVAILABLE:
        raise HTTPException(status_code=501, detail="Personalization system not available")
    
    try:
        await check_and_retrain_ranknet()
        
        # 학습 후 상태 확인
        is_trained = ranknet_recommender.is_trained
        
        return {
            "success": True,
            "ranknet_trained": is_trained,
            "message": "RankNet 학습 프로세스가 실행되었습니다."
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "RankNet 학습 실행 중 오류가 발생했습니다."
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 