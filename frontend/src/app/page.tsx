'use client';

import { useState } from 'react';
import SearchForm from './components/SearchForm';
import RecommendationResults from './components/RecommendationResults';

interface SearchQuery {
  keywords: string;
  priceMax: number | null;
  location: string;
  model: string;
  topK: number;
}

interface Recommendation {
  link: string;
  content: string;
  place: string;
  date: string;
  time: string;
  price_adv: number;
  price_door: number;
  score: number;
  model_used?: string;
}

interface ApiResponse {
  query: SearchQuery;
  recommendations: Recommendation[];
  total_count: number;
  model_used: string;
  available_models: string[];
}

export default function HomePage() {
  const [results, setResults] = useState<ApiResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (query: SearchQuery) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/recommend`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          keywords: query.keywords,
          price_max: query.priceMax,
          location: query.location,
          model: query.model,
          top_k: query.topK
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `서버 오류: ${response.status}`);
      }

      const data: ApiResponse = await response.json();
      setResults(data);
    } catch (err) {
      console.error('Search error:', err);
      setError(err instanceof Error ? err.message : '알 수 없는 오류가 발생했습니다.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50">
      {/* 헤더 */}
      <header className="bg-white/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <span className="text-white text-xl font-bold">🎵</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-800">
                  이벤트 추천 시스템
                </h1>
                <p className="text-sm text-gray-600">
                  AI 기반 다중 모델 추천 엔진
                </p>
              </div>
            </div>
            <div className="hidden md:flex items-center space-x-4">
              <div className="text-right">
                <div className="text-sm font-medium text-gray-700">Multi-Model AI</div>
                <div className="text-xs text-gray-500">TF-IDF • LSA • Word2Vec • Hybrid</div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* 메인 컨텐츠 */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* 검색 폼 */}
        <SearchForm onSearch={handleSearch} isLoading={isLoading} />

        {/* 오류 표시 */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-8">
            <div className="flex items-center space-x-2">
              <span className="text-red-500">❌</span>
              <span className="text-red-700 font-medium">오류 발생</span>
            </div>
            <p className="text-red-600 mt-2">{error}</p>
          </div>
        )}

        {/* 검색 결과 */}
        {results && (
          <div className="space-y-6">
            {/* 검색 요약 */}
            <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-gray-800">
                  검색 결과
                </h2>
                <div className="flex items-center space-x-4">
                  <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
                    {results.model_used.toUpperCase()} 모델
                  </span>
                  <span className="text-gray-600 text-sm">
                    총 {results.total_count}개 결과
                  </span>
                </div>
              </div>
              
              {/* 검색 조건 요약 */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">키워드:</span>
                  <span className="ml-2 font-medium">
                    {results.query.keywords || '전체'}
                  </span>
                </div>
                <div>
                  <span className="text-gray-500">지역:</span>
                  <span className="ml-2 font-medium">
                    {results.query.location || '전체'}
                  </span>
                </div>
                <div>
                  <span className="text-gray-500">최대 가격:</span>
                  <span className="ml-2 font-medium">
                    {results.query.priceMax ? `${results.query.priceMax.toLocaleString()}원` : '제한 없음'}
                  </span>
                </div>
                <div>
                  <span className="text-gray-500">추천 모델:</span>
                  <span className="ml-2 font-medium">
                    {results.model_used.toUpperCase()}
                  </span>
                </div>
              </div>
            </div>

            {/* 추천 결과 */}
            <RecommendationResults 
              recommendations={results.recommendations}
              searchQuery={results.query}
            />
          </div>
        )}

        {/* 초기 화면 안내 */}
        {!results && !isLoading && !error && (
          <div className="text-center py-16">
            <div className="w-24 h-24 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-6">
              <span className="text-white text-4xl">🎵</span>
            </div>
            <h2 className="text-2xl font-bold text-gray-800 mb-4">
              AI 기반 이벤트 추천 시스템
            </h2>
            <p className="text-gray-600 mb-8 max-w-2xl mx-auto">
              여러 AI 모델을 활용하여 당신에게 맞는 최적의 이벤트를 추천해드립니다. 
              키워드, 가격, 지역을 입력하고 원하는 추천 모델을 선택해보세요.
            </p>
            
            {/* 특징 소개 */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 max-w-4xl mx-auto">
              <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <span className="text-blue-600 text-xl">🔍</span>
                </div>
                <h3 className="font-semibold text-gray-800 mb-2">TF-IDF</h3>
                <p className="text-sm text-gray-600">키워드 빈도 기반 빠른 검색</p>
              </div>
              
              <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
                <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <span className="text-purple-600 text-xl">🧠</span>
                </div>
                <h3 className="font-semibold text-gray-800 mb-2">LSA</h3>
                <p className="text-sm text-gray-600">잠재 의미 분석으로 의미적 유사성 고려</p>
              </div>
              
              <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <span className="text-green-600 text-xl">🔗</span>
                </div>
                <h3 className="font-semibold text-gray-800 mb-2">Word2Vec</h3>
                <p className="text-sm text-gray-600">단어 임베딩으로 관계성 파악</p>
              </div>
              
              <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
                <div className="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <span className="text-orange-600 text-xl">🎯</span>
                </div>
                <h3 className="font-semibold text-gray-800 mb-2">Hybrid</h3>
                <p className="text-sm text-gray-600">여러 모델을 조합한 종합 추천</p>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* 푸터 */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="text-center text-gray-600">
            <p className="text-sm">
              © 2024 AI Event Recommendation System. 
              <span className="ml-2">Powered by Multi-Model Machine Learning</span>
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
