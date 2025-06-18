'use client';

import { useState, useEffect } from 'react';

interface SearchFormProps {
  onSearch: (query: {
    keywords: string;
    priceMax: number | null;
    location: string;
    model: string;
    topK: number;
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
    onSearch({
      keywords,
      priceMax: priceMax || null,
      location,
      model: selectedModel,
      topK
    });
  };

  const locationSuggestions = [
    '서울', '서울 강남구', '서울 강북구', '서울 강서구', '서울 관악구', 
    '서울 광진구', '서울 구로구', '서울 금천구', '서울 노원구', '서울 도봉구',
    '서울 동대문구', '서울 동작구', '서울 마포구', '서울 서대문구', '서울 서초구',
    '서울 성동구', '서울 성북구', '서울 송파구', '서울 양천구', '서울 영등포구',
    '서울 용산구', '서울 은평구', '서울 종로구', '서울 중구', '서울 중랑구',
    '서울 홍대', '서울 신촌', '서울 이태원', '서울 명동', '서울 강남역',
    '서울 신림', '서울 건대', '서울 잠실'
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
                  {model === 'lsa' && ' (의미 분석)'}
                  {model === 'word2vec' && ' (임베딩)'}
                  {model === 'hybrid' && ' (통합)'}
                </option>
              ))}
            </select>
            {/* 선택된 모델 설명 */}
            {modelInfo.model_descriptions[selectedModel] && (
              <p className="mt-2 text-xs text-gray-600">
                {modelInfo.model_descriptions[selectedModel]}
              </p>
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