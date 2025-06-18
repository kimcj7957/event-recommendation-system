from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from pathlib import Path

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

app = FastAPI(
    title="Multi-Model Event Recommendation API",
    description="이벤트 추천 시스템 - 다중 모델 지원 (TF-IDF, LSA, Word2Vec, Hybrid)",
    version="2.0.0"
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

class RecommendationResponse(BaseModel):
    query: dict
    recommendations: List[dict]
    total_count: int
    model_used: str
    available_models: List[str]

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
        print(f"✅ Available models: {multi_recommender.get_available_models()}")
    elif SINGLE_MODEL_AVAILABLE:
        print("⚠️  Running with single TF-IDF model only")
    else:
        print("❌ No recommendation models available")

@app.get("/")
async def root():
    return {
        "message": "Multi-Model Event Recommendation API",
        "version": "2.0.0",
        "multi_model_available": MULTI_MODEL_AVAILABLE,
        "single_model_available": SINGLE_MODEL_AVAILABLE
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "API is working properly",
        "multi_model_system": MULTI_MODEL_AVAILABLE,
        "single_model_fallback": SINGLE_MODEL_AVAILABLE
    }

@app.get("/models")
async def get_available_models():
    """사용 가능한 추천 모델 목록 반환"""
    if MULTI_MODEL_AVAILABLE and multi_recommender:
        available_models = multi_recommender.get_available_models()
    else:
        available_models = ["tfidf"] if SINGLE_MODEL_AVAILABLE else []
    
    return {
        "available_models": available_models,
        "model_descriptions": {
            "tfidf": "TF-IDF: 키워드 빈도 기반 추천 (빠름, 정확한 키워드 매칭)",
            "lsa": "LSA: 잠재 의미 분석 기반 추천 (의미적 유사성 고려)",
            "word2vec": "Word2Vec: 단어 임베딩 기반 추천 (단어 간 의미 관계 고려)",
            "hybrid": "Hybrid: 여러 모델 조합 추천 (종합적 결과)"
        }
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_events(query: RecommendationQuery):
    """이벤트 추천 API - 다중 모델 지원"""
    try:
        query_dict = {
            "keywords": query.keywords,
            "price_max": query.price_max,
            "location": query.location
        }
        
        # 다중 모델 시스템 사용
        if MULTI_MODEL_AVAILABLE and multi_recommender:
            available_models = multi_recommender.get_available_models()
            
            # 요청된 모델이 사용 가능한지 확인
            if query.model not in available_models:
                # 사용 가능한 모델 중 첫 번째로 대체
                actual_model = available_models[0] if available_models else "tfidf"
                print(f"Requested model '{query.model}' not available, using '{actual_model}'")
            else:
                actual_model = query.model
            
            # 추천 실행
            recommendations_df = multi_recommender.recommend(
                query_dict, 
                model_name=actual_model, 
                top_k=query.top_k
            )
            
            # DataFrame을 딕셔너리 리스트로 변환
            recommendations = recommendations_df.to_dict(orient='records')
            model_used = actual_model
            
        # 단일 모델 폴백
        elif SINGLE_MODEL_AVAILABLE:
            recommendations_df = single_recommend(query_dict, top_k=query.top_k)
            recommendations = recommendations_df.to_dict(orient='records')
            # 모델 사용 정보 추가
            for rec in recommendations:
                rec['model_used'] = 'tfidf'
            model_used = "tfidf"
            available_models = ["tfidf"]
            
        else:
            raise HTTPException(status_code=500, detail="No recommendation models available")
        
        # 응답 생성
        return RecommendationResponse(
            query=query_dict,
            recommendations=recommendations,
            total_count=len(recommendations),
            model_used=model_used,
            available_models=multi_recommender.get_available_models() if MULTI_MODEL_AVAILABLE else ["tfidf"]
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 