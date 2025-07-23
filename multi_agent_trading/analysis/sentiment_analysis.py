"""
Sentiment analysis module for news and social media data processing.
"""

import re
import string
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum


class SentimentPolarity(Enum):
    """Sentiment polarity enumeration."""
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


class NewsSource(Enum):
    """News source types."""
    FINANCIAL_NEWS = "FINANCIAL_NEWS"
    SOCIAL_MEDIA = "SOCIAL_MEDIA"
    ANALYST_REPORTS = "ANALYST_REPORTS"
    EARNINGS_CALLS = "EARNINGS_CALLS"
    REGULATORY_FILINGS = "REGULATORY_FILINGS"


@dataclass
class NewsData:
    """News data structure."""
    title: str
    content: str
    source: NewsSource
    timestamp: datetime
    symbol: Optional[str] = None
    author: Optional[str] = None
    url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "source": self.source.value,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "author": self.author,
            "url": self.url
        }


@dataclass
class SentimentScore:
    """Individual sentiment score for a piece of text."""
    polarity: SentimentPolarity
    confidence: float  # 0-1
    score: float  # -1 to 1, negative=bearish, positive=bullish
    keywords: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "polarity": self.polarity.value,
            "confidence": self.confidence,
            "score": self.score,
            "keywords": self.keywords
        }


@dataclass
class SentimentAnalysis:
    """Complete sentiment analysis result."""
    symbol: str
    timestamp: datetime
    overall_sentiment: SentimentPolarity
    overall_score: float  # -1 to 1
    overall_confidence: float  # 0-1
    news_sentiment: SentimentScore
    social_sentiment: SentimentScore
    analyst_sentiment: SentimentScore
    sentiment_trend: str  # IMPROVING, DECLINING, STABLE
    key_themes: List[str]
    risk_factors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "overall_sentiment": self.overall_sentiment.value,
            "overall_score": self.overall_score,
            "overall_confidence": self.overall_confidence,
            "news_sentiment": self.news_sentiment.to_dict(),
            "social_sentiment": self.social_sentiment.to_dict(),
            "analyst_sentiment": self.analyst_sentiment.to_dict(),
            "sentiment_trend": self.sentiment_trend,
            "key_themes": self.key_themes,
            "risk_factors": self.risk_factors
        }


class SentimentAnalyzer:
    """Sentiment analyzer for financial news and social media data."""
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        # Positive financial keywords
        self.positive_keywords = {
            'bullish', 'buy', 'strong', 'growth', 'profit', 'revenue', 'earnings',
            'beat', 'exceed', 'outperform', 'upgrade', 'positive', 'gain', 'rise',
            'increase', 'surge', 'rally', 'boom', 'expansion', 'success', 'record',
            'high', 'peak', 'breakthrough', 'innovation', 'opportunity', 'optimistic',
            'confident', 'momentum', 'upward', 'recovery', 'rebound', 'strength'
        }
        
        # Negative financial keywords
        self.negative_keywords = {
            'bearish', 'sell', 'weak', 'decline', 'loss', 'deficit', 'miss',
            'underperform', 'downgrade', 'negative', 'fall', 'drop', 'crash',
            'plunge', 'collapse', 'recession', 'crisis', 'risk', 'concern',
            'worry', 'fear', 'uncertainty', 'volatility', 'pressure', 'challenge',
            'problem', 'issue', 'warning', 'alert', 'caution', 'pessimistic',
            'doubt', 'struggle', 'difficulty', 'headwind', 'obstacle'
        }
        
        # Neutral/context keywords
        self.neutral_keywords = {
            'stable', 'maintain', 'hold', 'unchanged', 'flat', 'sideways',
            'consolidation', 'range', 'neutral', 'mixed', 'balanced'
        }
        
        # Risk-related keywords
        self.risk_keywords = {
            'risk', 'volatility', 'uncertainty', 'regulation', 'lawsuit',
            'investigation', 'fraud', 'scandal', 'bankruptcy', 'default',
            'debt', 'leverage', 'margin', 'liquidity', 'solvency'
        }
        
        # Theme keywords for categorization
        self.theme_keywords = {
            'earnings': {'earnings', 'revenue', 'profit', 'eps', 'guidance'},
            'merger': {'merger', 'acquisition', 'takeover', 'deal', 'buyout'},
            'product': {'product', 'launch', 'innovation', 'technology', 'patent'},
            'management': {'ceo', 'management', 'leadership', 'executive', 'board'},
            'regulatory': {'regulation', 'fda', 'approval', 'compliance', 'policy'},
            'market': {'market', 'competition', 'share', 'demand', 'supply'}
        }
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned and preprocessed text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove punctuation for keyword matching
        text_no_punct = text.translate(str.maketrans('', '', string.punctuation))
        
        return text_no_punct
    
    def extract_keywords(self, text: str) -> Tuple[List[str], List[str], List[str]]:
        """
        Extract positive, negative, and risk keywords from text.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Tuple of (positive_keywords, negative_keywords, risk_keywords)
        """
        words = set(text.split())
        
        positive_found = list(words.intersection(self.positive_keywords))
        negative_found = list(words.intersection(self.negative_keywords))
        risk_found = list(words.intersection(self.risk_keywords))
        
        return positive_found, negative_found, risk_found
    
    def calculate_sentiment_score(self, text: str) -> SentimentScore:
        """
        Calculate sentiment score for a piece of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score with polarity, confidence, and keywords
        """
        if not text:
            return SentimentScore(
                polarity=SentimentPolarity.NEUTRAL,
                confidence=0.0,
                score=0.0,
                keywords=[]
            )
        
        processed_text = self.preprocess_text(text)
        positive_keywords, negative_keywords, risk_keywords = self.extract_keywords(processed_text)
        
        # Calculate raw scores
        positive_score = len(positive_keywords)
        negative_score = len(negative_keywords)
        risk_score = len(risk_keywords)
        
        # Apply risk penalty
        adjusted_positive = max(0, positive_score - risk_score * 0.5)
        adjusted_negative = negative_score + risk_score * 0.3
        
        # Calculate final score (-1 to 1)
        total_keywords = adjusted_positive + adjusted_negative
        if total_keywords == 0:
            final_score = 0.0
            polarity = SentimentPolarity.NEUTRAL
            confidence = 0.1  # Low confidence for neutral
        else:
            final_score = (adjusted_positive - adjusted_negative) / total_keywords
            
            # Determine polarity
            if final_score > 0.1:
                polarity = SentimentPolarity.POSITIVE
            elif final_score < -0.1:
                polarity = SentimentPolarity.NEGATIVE
            else:
                polarity = SentimentPolarity.NEUTRAL
            
            # Calculate confidence based on keyword count and text length
            word_count = len(processed_text.split())
            keyword_density = total_keywords / max(word_count, 1)
            confidence = min(1.0, keyword_density * 10 + 0.3)  # Base confidence + density bonus
        
        # Combine all found keywords
        all_keywords = positive_keywords + negative_keywords + risk_keywords
        
        return SentimentScore(
            polarity=polarity,
            confidence=confidence,
            score=final_score,
            keywords=all_keywords
        )
    
    def extract_themes(self, text: str) -> List[str]:
        """
        Extract key themes from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of identified themes
        """
        processed_text = self.preprocess_text(text)
        words = set(processed_text.split())
        
        themes = []
        for theme, keywords in self.theme_keywords.items():
            if words.intersection(keywords):
                themes.append(theme)
        
        return themes
    
    def analyze_news_sentiment(self, news_items: List[NewsData]) -> SentimentScore:
        """
        Analyze sentiment from news data.
        
        Args:
            news_items: List of news items to analyze
            
        Returns:
            Aggregated sentiment score for news
        """
        if not news_items:
            return SentimentScore(
                polarity=SentimentPolarity.NEUTRAL,
                confidence=0.0,
                score=0.0,
                keywords=[]
            )
        
        scores = []
        all_keywords = []
        
        for news in news_items:
            # Combine title and content for analysis
            full_text = f"{news.title} {news.content}"
            score = self.calculate_sentiment_score(full_text)
            scores.append(score)
            all_keywords.extend(score.keywords)
        
        # Calculate weighted average (more recent news has higher weight)
        total_score = 0.0
        total_weight = 0.0
        
        for i, score in enumerate(scores):
            # Weight decreases with age (assuming news_items are sorted by time)
            weight = 1.0 / (i + 1)
            total_score += score.score * weight * score.confidence
            total_weight += weight * score.confidence
        
        if total_weight == 0:
            final_score = 0.0
            confidence = 0.0
        else:
            final_score = total_score / total_weight
            confidence = min(1.0, total_weight / len(scores))
        
        # Determine overall polarity
        if final_score > 0.1:
            polarity = SentimentPolarity.POSITIVE
        elif final_score < -0.1:
            polarity = SentimentPolarity.NEGATIVE
        else:
            polarity = SentimentPolarity.NEUTRAL
        
        return SentimentScore(
            polarity=polarity,
            confidence=confidence,
            score=final_score,
            keywords=list(set(all_keywords))  # Remove duplicates
        )
    
    def analyze_social_sentiment(self, social_data: List[str]) -> SentimentScore:
        """
        Analyze sentiment from social media data.
        
        Args:
            social_data: List of social media posts/comments
            
        Returns:
            Aggregated sentiment score for social media
        """
        if not social_data:
            return SentimentScore(
                polarity=SentimentPolarity.NEUTRAL,
                confidence=0.0,
                score=0.0,
                keywords=[]
            )
        
        scores = []
        all_keywords = []
        
        for post in social_data:
            score = self.calculate_sentiment_score(post)
            scores.append(score)
            all_keywords.extend(score.keywords)
        
        # Simple average for social media (all posts weighted equally)
        if scores:
            avg_score = sum(s.score for s in scores) / len(scores)
            avg_confidence = sum(s.confidence for s in scores) / len(scores)
        else:
            avg_score = 0.0
            avg_confidence = 0.0
        
        # Determine polarity
        if avg_score > 0.1:
            polarity = SentimentPolarity.POSITIVE
        elif avg_score < -0.1:
            polarity = SentimentPolarity.NEGATIVE
        else:
            polarity = SentimentPolarity.NEUTRAL
        
        return SentimentScore(
            polarity=polarity,
            confidence=avg_confidence,
            score=avg_score,
            keywords=list(set(all_keywords))
        )
    
    def detect_sentiment_trend(self, historical_scores: List[float]) -> str:
        """
        Detect sentiment trend from historical scores.
        
        Args:
            historical_scores: List of historical sentiment scores (oldest first)
            
        Returns:
            Trend direction: IMPROVING, DECLINING, or STABLE
        """
        if len(historical_scores) < 2:
            return "STABLE"
        
        # Calculate trend using simple linear regression slope
        n = len(historical_scores)
        x_values = list(range(n))
        
        # Calculate slope
        x_mean = sum(x_values) / n
        y_mean = sum(historical_scores) / n
        
        numerator = sum((x_values[i] - x_mean) * (historical_scores[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "STABLE"
        
        slope = numerator / denominator
        
        # Determine trend based on slope
        if slope > 0.05:
            return "IMPROVING"
        elif slope < -0.05:
            return "DECLINING"
        else:
            return "STABLE"
    
    def analyze(self, symbol: str, news_data: List[NewsData], 
                social_data: List[str], analyst_reports: List[str],
                historical_sentiment: List[float] = None) -> SentimentAnalysis:
        """
        Perform complete sentiment analysis.
        
        Args:
            symbol: Trading symbol
            news_data: List of news items
            social_data: List of social media posts
            analyst_reports: List of analyst report texts
            historical_sentiment: Historical sentiment scores for trend analysis
            
        Returns:
            Complete sentiment analysis
        """
        # Analyze different data sources
        news_sentiment = self.analyze_news_sentiment(news_data)
        social_sentiment = self.analyze_social_sentiment(social_data)
        
        # Analyze analyst reports (treat similar to news)
        analyst_news = [
            NewsData(
                title="Analyst Report",
                content=report,
                source=NewsSource.ANALYST_REPORTS,
                timestamp=datetime.utcnow()
            ) for report in analyst_reports
        ]
        analyst_sentiment = self.analyze_news_sentiment(analyst_news)
        
        # Calculate overall sentiment (weighted average)
        news_weight = 0.4
        social_weight = 0.3
        analyst_weight = 0.3
        
        overall_score = (
            news_sentiment.score * news_weight * news_sentiment.confidence +
            social_sentiment.score * social_weight * social_sentiment.confidence +
            analyst_sentiment.score * analyst_weight * analyst_sentiment.confidence
        )
        
        total_weight = (
            news_weight * news_sentiment.confidence +
            social_weight * social_sentiment.confidence +
            analyst_weight * analyst_sentiment.confidence
        )
        
        if total_weight > 0:
            overall_score = overall_score / total_weight
            overall_confidence = total_weight / (news_weight + social_weight + analyst_weight)
        else:
            overall_score = 0.0
            overall_confidence = 0.0
        
        # Determine overall polarity
        if overall_score > 0.1:
            overall_sentiment = SentimentPolarity.POSITIVE
        elif overall_score < -0.1:
            overall_sentiment = SentimentPolarity.NEGATIVE
        else:
            overall_sentiment = SentimentPolarity.NEUTRAL
        
        # Extract themes from all text
        all_text = ""
        for news in news_data:
            all_text += f" {news.title} {news.content}"
        for post in social_data:
            all_text += f" {post}"
        for report in analyst_reports:
            all_text += f" {report}"
        
        key_themes = self.extract_themes(all_text)
        
        # Extract risk factors
        risk_keywords = []
        for news in news_data:
            _, _, risks = self.extract_keywords(self.preprocess_text(f"{news.title} {news.content}"))
            risk_keywords.extend(risks)
        
        risk_factors = list(set(risk_keywords))
        
        # Detect sentiment trend
        if historical_sentiment:
            sentiment_trend = self.detect_sentiment_trend(historical_sentiment + [overall_score])
        else:
            sentiment_trend = "STABLE"
        
        return SentimentAnalysis(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            overall_sentiment=overall_sentiment,
            overall_score=overall_score,
            overall_confidence=overall_confidence,
            news_sentiment=news_sentiment,
            social_sentiment=social_sentiment,
            analyst_sentiment=analyst_sentiment,
            sentiment_trend=sentiment_trend,
            key_themes=key_themes,
            risk_factors=risk_factors
        )