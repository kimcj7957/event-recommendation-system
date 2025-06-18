'use client';

import { useState } from 'react';

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

interface SearchQuery {
  keywords: string;
  priceMax: number | null;
  location: string;
  model: string;
  topK: number;
}

interface RecommendationResultsProps {
  recommendations: Recommendation[];
  searchQuery: SearchQuery;
}

export default function RecommendationResults({ recommendations, searchQuery }: RecommendationResultsProps) {
  const [likedEvents, setLikedEvents] = useState<Set<string>>(new Set());

  const toggleLike = (eventLink: string) => {
    setLikedEvents(prev => {
      const newSet = new Set(prev);
      if (newSet.has(eventLink)) {
        newSet.delete(eventLink);
      } else {
        newSet.add(eventLink);
      }
      return newSet;
    });
  };

  if (recommendations.length === 0) {
    return (
      <div className="text-center py-12 bg-white rounded-xl shadow-lg">
        <div className="text-6xl mb-4">🎭</div>
        <h3 className="text-xl font-semibold text-gray-700 mb-2">검색 결과가 없습니다</h3>
        <p className="text-gray-500 mb-4">다른 키워드나 조건으로 다시 검색해보세요!</p>
        <div className="text-sm text-gray-400">
          <p>💡 팁: 더 일반적인 키워드를 사용하거나 가격 범위를 늘려보세요</p>
        </div>
      </div>
    );
  }

  const formatDate = (dateString: string) => {
    if (dateString === 'NaT' || !dateString || dateString === 'null' || dateString === 'unknown') {
      return '날짜 미정';
    }
    
    try {
      const date = new Date(dateString);
      if (isNaN(date.getTime())) {
        return '날짜 미정';
      }
      return date.toLocaleDateString('ko-KR', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        weekday: 'short'
      });
    } catch {
      return '날짜 미정';
    }
  };

  const formatTime = (timeString: string) => {
    if (timeString === 'NaT' || !timeString || timeString === 'null' || timeString === 'unknown') {
      return '';
    }
    return timeString;
  };

  const formatPrice = (price: number) => {
    if (!price || price === 0) return '무료';
    return new Intl.NumberFormat('ko-KR').format(price);
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-emerald-700 bg-emerald-100 border-emerald-200';
    if (score >= 0.6) return 'text-blue-700 bg-blue-100 border-blue-200';
    if (score >= 0.4) return 'text-amber-700 bg-amber-100 border-amber-200';
    return 'text-gray-700 bg-gray-100 border-gray-200';
  };

  const getModelBadgeColor = (model: string) => {
    switch (model?.toLowerCase()) {
      case 'tfidf': return 'bg-blue-500';
      case 'lsa': return 'bg-purple-500';
      case 'word2vec': return 'bg-green-500';
      case 'hybrid': return 'bg-gradient-to-r from-blue-500 to-purple-500';
      default: return 'bg-gray-500';
    }
  };

  return (
    <div className="space-y-6">
      {/* 결과 헤더 */}
      <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-bold text-gray-800">
            🎯 맞춤 추천 결과
          </h2>
          <div className="text-sm text-gray-600">
            총 {recommendations.length}개의 이벤트
          </div>
        </div>
      </div>

      {/* 추천 카드들 */}
      <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
        {recommendations.map((event, index) => (
          <div
            key={`${event.link}-${index}`}
            className="bg-white rounded-xl shadow-sm hover:shadow-lg transition-all duration-300 overflow-hidden border border-gray-200 group hover:border-blue-300"
          >
            <div className="p-6">
              {/* 헤더: 순위, 점수, 모델 */}
              <div className="flex justify-between items-start mb-4">
                <div className="flex items-center space-x-2">
                  <span className="bg-gradient-to-r from-blue-500 to-purple-600 text-white text-sm font-bold px-3 py-1 rounded-full">
                    #{index + 1}
                  </span>
                  <span className={`text-xs font-medium px-2 py-1 rounded-full border ${getScoreColor(event.score)}`}>
                    {Math.round(event.score * 100)}% 매치
                  </span>
                </div>
                {event.model_used && (
                  <span className={`text-xs text-white px-2 py-1 rounded-full ${getModelBadgeColor(event.model_used)}`}>
                    {event.model_used.toUpperCase()}
                  </span>
                )}
              </div>

              {/* 이벤트 제목/내용 */}
              <div className="mb-4">
                <h3 className="font-semibold text-gray-800 mb-2 line-clamp-3 group-hover:text-blue-600 transition-colors leading-snug">
                  {event.content.length > 100 
                    ? event.content.substring(0, 100) + '...' 
                    : event.content}
                </h3>
              </div>

              {/* 메타 정보들 */}
              <div className="space-y-3 mb-4">
                {/* 장소 */}
                <div className="flex items-start text-sm text-gray-600">
                  <span className="mr-2 mt-0.5 flex-shrink-0">📍</span>
                  <span className="line-clamp-2">{event.place || '장소 미정'}</span>
                </div>

                {/* 날짜 및 시간 */}
                <div className="flex items-center text-sm text-gray-600">
                  <span className="mr-2">📅</span>
                  <span>{formatDate(event.date)}</span>
                  {formatTime(event.time) && (
                    <>
                      <span className="mx-2 text-gray-400">•</span>
                      <span className="flex items-center">
                        ⏰ {formatTime(event.time)}
                      </span>
                    </>
                  )}
                </div>

                {/* 가격 정보 */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center text-sm">
                    <span className="mr-1">💰</span>
                    <span className="font-medium text-blue-600">
                      예매: {formatPrice(event.price_adv)}원
                    </span>
                  </div>
                  {event.price_door !== event.price_adv && event.price_door > 0 && (
                    <div className="text-xs text-gray-500">
                      현장: {formatPrice(event.price_door)}원
                    </div>
                  )}
                </div>
              </div>

              {/* 링크 버튼 및 좋아요 버튼 */}
              <div className="pt-2 border-t border-gray-100 space-y-3">
                {/* 좋아요 버튼 */}
                <div className="flex items-center justify-center">
                  <button
                    onClick={() => toggleLike(event.link)}
                    className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                      likedEvents.has(event.link)
                        ? 'bg-pink-50 text-pink-600 border border-pink-200 hover:bg-pink-100'
                        : 'bg-gray-50 text-gray-600 border border-gray-200 hover:bg-gray-100'
                    }`}
                  >
                    <span className={`text-lg ${likedEvents.has(event.link) ? 'animate-pulse' : ''}`}>
                      {likedEvents.has(event.link) ? '❤️' : '🤍'}
                    </span>
                    <span className="text-sm font-medium">
                      {likedEvents.has(event.link) ? '관심 이벤트' : '관심 추가'}
                    </span>
                  </button>
                </div>
                
                {/* 링크 버튼 */}
                <a
                  href={event.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block w-full text-center bg-gradient-to-r from-blue-500 to-purple-600 text-white py-3 px-4 rounded-lg hover:from-blue-600 hover:to-purple-700 transition-all duration-200 text-sm font-medium hover:shadow-md transform hover:-translate-y-0.5"
                >
                  🔗 자세히 보기
                </a>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* 추가 정보 푸터 */}
      <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
        <div className="text-center text-sm text-gray-600">
          <p className="mb-2">
            💡 <strong>추천 점수</strong>는 검색 조건과의 유사도를 나타냅니다
          </p>
          <p className="text-xs text-gray-500">
            더 정확한 추천을 위해 구체적인 키워드나 지역을 입력해보세요
          </p>
        </div>
      </div>
    </div>
  );
}