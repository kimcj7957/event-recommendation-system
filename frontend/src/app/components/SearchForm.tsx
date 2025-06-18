'use client';

import { useState, useEffect } from 'react';

interface SearchFormProps {
  onSearch: (query: {
    keywords: string;
    priceMax: number | null;
    location: string;
    model: string;
    topK: number;
    userId: string;
  }) => void;
  isLoading: boolean;
}

interface ModelInfo {
  available_models: string[];
  model_descriptions: Record<string, string>;
}

export default function SearchForm({ onSearch, isLoading }: SearchFormProps) {
  const [keywords, setKeywords] = useState('');
  const [priceMax, setPriceMax] = useState<number | null>(null);
  const [location, setLocation] = useState('');
  const [selectedModel, setSelectedModel] = useState('tfidf');
  const [topK, setTopK] = useState(5);
  const [modelInfo, setModelInfo] = useState<ModelInfo>({
    available_models: ['tfidf'],
    model_descriptions: {}
  });

  // 사용자 ID 가져오기
  const getUserId = () => {
    if (typeof window !== 'undefined') {
      let userId = localStorage.getItem('event_user_id');
      if (!userId) {
        userId = 'user_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('event_user_id', userId);
      }
      return userId;
    }
    return null;
  };

  // 사용 가능한 모델 정보 로드
  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/models`);
        if (response.ok) {
          const data = await response.json();
          setModelInfo(data);
          // 첫 번째 사용 가능한 모델을 기본값으로 설정
          if (data.available_models.length > 0) {
            setSelectedModel(data.available_models[0]);
          }
        }
      } catch (error) {
        console.error('Failed to fetch model info:', error);
      }
    };

    fetchModelInfo();
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const userId = getUserId();
    
    onSearch({
      keywords,
      priceMax,
      location,
      model: selectedModel,
      topK,
      userId: userId || 'anonymous'  // null 처리
    });
  };

  const locationSuggestions = [

    
    // 세부 지역명
    '서울 중구', '서울 마포구',
    // 실제 데이터에서 가장 많이 사용되는 공연장들 (TOP 10)
    '서교동 407-8 B1, Seoul, Korea ClubFF',
    '홍대 언플러그드',
    '우주정거장', 
    '서울특별시 마포구 양화로 12길 6',
    '스트레인지프룻',
    '생기스튜디오',
    '신도시',
    'Club Victim',
    '서울 마포구 양화로6길 27 지하 1층',
    '무대륙',
    '서울 중구 수표로6길 10 지하1층'
  ];

  return (
    <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
      <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
        🎵 이벤트 추천 검색
        <span className="ml-3 text-sm font-normal text-blue-600 bg-blue-50 px-3 py-1 rounded-full">
          다중 모델 지원
        </span>
      </h2>
      
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* 키워드 검색 */}
        <div>
          <label htmlFor="keywords" className="block text-sm font-medium text-gray-700 mb-2">
            🔍 키워드
          </label>
          <input
            type="text"
            id="keywords"
            value={keywords}
            onChange={(e) => setKeywords(e.target.value)}
            placeholder="예: 재즈, 록 콘서트, 클래식, 힙합..."
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-900"
          />
        </div>

        {/* 가격 및 지역을 한 줄에 배치 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* 최대 가격 */}
          <div>
            <label htmlFor="priceMax" className="block text-sm font-medium text-gray-700 mb-2">
              💰 최대 가격 (원)
            </label>
            <input
              type="number"
              id="priceMax"
              value={priceMax || ''}
              onChange={(e) => setPriceMax(e.target.value ? parseInt(e.target.value) : null)}
              placeholder="예: 50000"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-900"
            />
          </div>

          {/* 지역 */}
          <div>
            <label htmlFor="location" className="block text-sm font-medium text-gray-700 mb-2">
              📍 지역
            </label>
            <select
              id="location"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-900"
            >
              <option value="">전체 지역</option>
              {locationSuggestions.map((loc) => (
                <option key={loc} value={loc}>{loc}</option>
              ))}
            </select>
          </div>
        </div>

        {/* 모델 선택 및 결과 개수를 한 줄에 배치 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* 추천 모델 선택 */}
          <div>
            <label htmlFor="model" className="block text-sm font-medium text-gray-700 mb-2">
              🤖 추천 모델
            </label>
            <select
              id="model"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-900"
            >
              {modelInfo.available_models.map((model) => (
                <option key={model} value={model}>
                  {model.toUpperCase()}
                  {model === 'tfidf' && ' (기본)'}
                  {model === 'ranknet' && ' (개인화)'}
                  {model === 'balanced' && ' (균등가중치)'}
                </option>
              ))}
            </select>
            {/* 선택된 모델 설명 */}
            {modelInfo.model_descriptions[selectedModel] && (
              <p className="mt-2 text-xs text-gray-600">
                💡 {modelInfo.model_descriptions[selectedModel]}
              </p>
            )}
            
            {/* RankNet 모델 특별 안내 */}
            {selectedModel === 'ranknet' && (
              <div className="mt-2 p-3 bg-pink-50 border border-pink-200 rounded-lg">
                <p className="text-xs text-pink-700">
                  🎯 <strong>개인화 추천</strong>: 당신의 좋아요 패턴을 학습하여 맞춤형 이벤트를 추천합니다.
                  <br />
                  💡 더 많은 좋아요를 누를수록 추천이 정확해집니다!
                </p>
              </div>
            )}
            
            {/* Balanced 모델 특별 안내 */}
            {selectedModel === 'balanced' && (
              <div className="mt-2 p-3 bg-green-50 border border-green-200 rounded-lg">
                <p className="text-xs text-green-700">
                  ⚖️ <strong>균등 가중치 추천</strong>: 내용, 가격, 위치를 균등하게 고려하여 추천합니다.
                  <br />
                  📊 내용 34% + 가격 33% + 위치 33% = 균형잡힌 추천
                </p>
              </div>
            )}
          </div>

          {/* 결과 개수 */}
          <div>
            <label htmlFor="topK" className="block text-sm font-medium text-gray-700 mb-2">
              📊 결과 개수
            </label>
            <select
              id="topK"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-900"
            >
              <option value={3}>3개</option>
              <option value={5}>5개 (추천)</option>
              <option value={10}>10개</option>
              <option value={15}>15개</option>
              <option value={20}>20개</option>
            </select>
          </div>
        </div>

        {/* 검색 버튼 */}
        <button
          type="submit"
          disabled={isLoading}
          className={`w-full py-4 px-6 rounded-lg font-semibold text-white transition-all duration-200 ${
            isLoading
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 hover:shadow-lg transform hover:-translate-y-0.5'
          }`}
        >
          {isLoading ? (
            <div className="flex items-center justify-center">
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
              추천 검색 중...
            </div>
          ) : (
            '🚀 이벤트 추천받기'
          )}
        </button>
      </form>

      {/* 모델 정보 카드 */}
      {modelInfo.available_models.length > 1 && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <h3 className="text-sm font-semibold text-gray-700 mb-2">💡 모델 설명</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {Object.entries(modelInfo.model_descriptions).map(([model, description]) => (
              <div key={model} className="text-xs text-gray-600">
                <strong className="text-blue-600">{model.toUpperCase()}:</strong> {description}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
} 