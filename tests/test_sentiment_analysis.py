"""
Unit tests for sentiment analysis module.
"""

import pytest
from datetime import datetime
from multi_agent_trading.analysis.sentiment_analysis import (
    SentimentAnalyzer, SentimentScore, SentimentAnalysis, NewsData,
    SentimentPolarity, NewsSource
)


class TestSentimentAnalyzer:
    """Test cases for SentimentAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
        
        # Sample news data
        self.positive_news = NewsData(
            title="Company Reports Record Earnings Beat",
            content="The company exceeded analyst expectations with strong revenue growth and bullish outlook for next quarter. Management expressed confidence in continued expansion.",
            source=NewsSource.FINANCIAL_NEWS,
            timestamp=datetime.utcnow(),
            symbol="AAPL"
        )
        
        self.negative_news = NewsData(
            title="Stock Plunges on Weak Guidance",
            content="Shares fell sharply after the company warned of declining demand and increased competition. Analysts downgraded the stock citing concerns about future profitability.",
            source=NewsSource.FINANCIAL_NEWS,
            timestamp=datetime.utcnow(),
            symbol="AAPL"
        )
        
        self.neutral_news = NewsData(
            title="Company Maintains Stable Operations",
            content="The quarterly report showed unchanged revenue and flat earnings. Management indicated they will hold current positions and maintain existing strategies.",
            source=NewsSource.FINANCIAL_NEWS,
            timestamp=datetime.utcnow(),
            symbol="AAPL"
        )
        
        # Sample social media data
        self.positive_social = [
            "AAPL looking strong! Great earnings beat, buying more shares",
            "Bullish on Apple, innovation continues to drive growth",
            "Record profits show this company is unstoppable"
        ]
        
        self.negative_social = [
            "AAPL is overvalued, expecting a crash soon",
            "Weak guidance and declining sales, time to sell",
            "Too much risk in this market, bearish on tech stocks"
        ]
        
        self.neutral_social = [
            "AAPL trading sideways, waiting for direction",
            "Mixed signals from the market, holding position",
            "Stable company but no clear catalyst"
        ]
    
    def test_preprocess_text_basic(self):
        """Test basic text preprocessing."""
        text = "This is a TEST with CAPS and numbers 123!"
        processed = self.analyzer.preprocess_text(text)
        
        assert processed == "this is a test with caps and numbers 123"
    
    def test_preprocess_text_urls_and_emails(self):
        """Test preprocessing removes URLs and emails."""
        text = "Check out https://example.com and email me at test@example.com"
        processed = self.analyzer.preprocess_text(text)
        
        assert "https://example.com" not in processed
        assert "test@example.com" not in processed
        assert "check out" in processed
        assert "and email me at" in processed
    
    def test_preprocess_text_special_characters(self):
        """Test preprocessing handles special characters."""
        text = "Stock $AAPL @mention #hashtag & other symbols!"
        processed = self.analyzer.preprocess_text(text)
        
        # Should remove most special characters but keep basic structure
        assert "$" not in processed
        assert "@" not in processed
        assert "#" not in processed
        assert "stock" in processed
        assert "aapl" in processed
    
    def test_extract_keywords_positive(self):
        """Test extraction of positive keywords."""
        text = "strong growth profit bullish buy upgrade"
        positive, negative, risk = self.analyzer.extract_keywords(text)
        
        assert len(positive) > 0
        assert "strong" in positive
        assert "growth" in positive
        assert "profit" in positive
        assert "bullish" in positive
        assert len(negative) == 0
    
    def test_extract_keywords_negative(self):
        """Test extraction of negative keywords."""
        text = "weak decline loss bearish sell downgrade"
        positive, negative, risk = self.analyzer.extract_keywords(text)
        
        assert len(negative) > 0
        assert "weak" in negative
        assert "decline" in negative
        assert "loss" in negative
        assert "bearish" in negative
        assert len(positive) == 0
    
    def test_extract_keywords_risk(self):
        """Test extraction of risk keywords."""
        text = "volatility uncertainty risk regulation lawsuit"
        positive, negative, risk = self.analyzer.extract_keywords(text)
        
        assert len(risk) > 0
        assert "volatility" in risk
        assert "uncertainty" in risk
        assert "risk" in risk
    
    def test_calculate_sentiment_score_positive(self):
        """Test sentiment score calculation for positive text."""
        text = "excellent earnings beat strong growth bullish outlook"
        score = self.analyzer.calculate_sentiment_score(text)
        
        assert score.polarity == SentimentPolarity.POSITIVE
        assert score.score > 0
        assert score.confidence > 0
        assert len(score.keywords) > 0
    
    def test_calculate_sentiment_score_negative(self):
        """Test sentiment score calculation for negative text."""
        text = "terrible earnings miss weak performance bearish outlook"
        score = self.analyzer.calculate_sentiment_score(text)
        
        assert score.polarity == SentimentPolarity.NEGATIVE
        assert score.score < 0
        assert score.confidence > 0
        assert len(score.keywords) > 0
    
    def test_calculate_sentiment_score_neutral(self):
        """Test sentiment score calculation for neutral text."""
        text = "the company reported quarterly results as expected"
        score = self.analyzer.calculate_sentiment_score(text)
        
        assert score.polarity == SentimentPolarity.NEUTRAL
        assert abs(score.score) < 0.1
        assert score.confidence >= 0
    
    def test_calculate_sentiment_score_empty(self):
        """Test sentiment score calculation for empty text."""
        score = self.analyzer.calculate_sentiment_score("")
        
        assert score.polarity == SentimentPolarity.NEUTRAL
        assert score.score == 0.0
        assert score.confidence == 0.0
        assert len(score.keywords) == 0
    
    def test_calculate_sentiment_score_risk_penalty(self):
        """Test that risk keywords reduce positive sentiment."""
        positive_text = "strong growth profit excellent"
        risky_text = "strong growth profit excellent but high risk volatility uncertainty"
        
        positive_score = self.analyzer.calculate_sentiment_score(positive_text)
        risky_score = self.analyzer.calculate_sentiment_score(risky_text)
        
        # Risk should reduce the positive sentiment
        assert risky_score.score < positive_score.score
    
    def test_extract_themes_earnings(self):
        """Test theme extraction for earnings-related content."""
        text = "quarterly earnings revenue profit eps guidance"
        themes = self.analyzer.extract_themes(text)
        
        assert "earnings" in themes
    
    def test_extract_themes_merger(self):
        """Test theme extraction for merger-related content."""
        text = "acquisition merger takeover deal buyout"
        themes = self.analyzer.extract_themes(text)
        
        assert "merger" in themes
    
    def test_extract_themes_multiple(self):
        """Test extraction of multiple themes."""
        text = "earnings beat merger announcement new product launch"
        themes = self.analyzer.extract_themes(text)
        
        assert len(themes) >= 2
        assert "earnings" in themes
        assert "product" in themes
    
    def test_analyze_news_sentiment_positive(self):
        """Test news sentiment analysis with positive news."""
        news_list = [self.positive_news]
        sentiment = self.analyzer.analyze_news_sentiment(news_list)
        
        assert sentiment.polarity == SentimentPolarity.POSITIVE
        assert sentiment.score > 0
        assert sentiment.confidence > 0
        assert len(sentiment.keywords) > 0
    
    def test_analyze_news_sentiment_negative(self):
        """Test news sentiment analysis with negative news."""
        news_list = [self.negative_news]
        sentiment = self.analyzer.analyze_news_sentiment(news_list)
        
        assert sentiment.polarity == SentimentPolarity.NEGATIVE
        assert sentiment.score < 0
        assert sentiment.confidence > 0
    
    def test_analyze_news_sentiment_mixed(self):
        """Test news sentiment analysis with mixed news."""
        news_list = [self.positive_news, self.negative_news, self.neutral_news]
        sentiment = self.analyzer.analyze_news_sentiment(news_list)
        
        # Should be somewhere in between
        assert isinstance(sentiment.polarity, SentimentPolarity)
        assert -1 <= sentiment.score <= 1
        assert 0 <= sentiment.confidence <= 1
    
    def test_analyze_news_sentiment_empty(self):
        """Test news sentiment analysis with empty list."""
        sentiment = self.analyzer.analyze_news_sentiment([])
        
        assert sentiment.polarity == SentimentPolarity.NEUTRAL
        assert sentiment.score == 0.0
        assert sentiment.confidence == 0.0
        assert len(sentiment.keywords) == 0
    
    def test_analyze_social_sentiment_positive(self):
        """Test social media sentiment analysis with positive posts."""
        sentiment = self.analyzer.analyze_social_sentiment(self.positive_social)
        
        assert sentiment.polarity == SentimentPolarity.POSITIVE
        assert sentiment.score > 0
        assert sentiment.confidence > 0
    
    def test_analyze_social_sentiment_negative(self):
        """Test social media sentiment analysis with negative posts."""
        sentiment = self.analyzer.analyze_social_sentiment(self.negative_social)
        
        assert sentiment.polarity == SentimentPolarity.NEGATIVE
        assert sentiment.score < 0
        assert sentiment.confidence > 0
    
    def test_analyze_social_sentiment_neutral(self):
        """Test social media sentiment analysis with neutral posts."""
        sentiment = self.analyzer.analyze_social_sentiment(self.neutral_social)
        
        assert sentiment.polarity == SentimentPolarity.NEUTRAL
        assert abs(sentiment.score) <= 0.1
    
    def test_analyze_social_sentiment_empty(self):
        """Test social media sentiment analysis with empty list."""
        sentiment = self.analyzer.analyze_social_sentiment([])
        
        assert sentiment.polarity == SentimentPolarity.NEUTRAL
        assert sentiment.score == 0.0
        assert sentiment.confidence == 0.0
    
    def test_detect_sentiment_trend_improving(self):
        """Test sentiment trend detection for improving sentiment."""
        scores = [-0.5, -0.3, -0.1, 0.1, 0.3]
        trend = self.analyzer.detect_sentiment_trend(scores)
        
        assert trend == "IMPROVING"
    
    def test_detect_sentiment_trend_declining(self):
        """Test sentiment trend detection for declining sentiment."""
        scores = [0.5, 0.3, 0.1, -0.1, -0.3]
        trend = self.analyzer.detect_sentiment_trend(scores)
        
        assert trend == "DECLINING"
    
    def test_detect_sentiment_trend_stable(self):
        """Test sentiment trend detection for stable sentiment."""
        scores = [0.1, 0.12, 0.08, 0.11, 0.09]
        trend = self.analyzer.detect_sentiment_trend(scores)
        
        assert trend == "STABLE"
    
    def test_detect_sentiment_trend_insufficient_data(self):
        """Test sentiment trend detection with insufficient data."""
        scores = [0.1]
        trend = self.analyzer.detect_sentiment_trend(scores)
        
        assert trend == "STABLE"
    
    def test_analyze_complete_positive(self):
        """Test complete sentiment analysis with positive data."""
        analysis = self.analyzer.analyze(
            symbol="AAPL",
            news_data=[self.positive_news],
            social_data=self.positive_social,
            analyst_reports=["Strong buy recommendation with price target increase"],
            historical_sentiment=[0.1, 0.2, 0.3]
        )
        
        # Verify analysis structure
        assert isinstance(analysis, SentimentAnalysis)
        assert analysis.symbol == "AAPL"
        assert isinstance(analysis.timestamp, datetime)
        assert analysis.overall_sentiment == SentimentPolarity.POSITIVE
        assert analysis.overall_score > 0
        assert 0 <= analysis.overall_confidence <= 1
        assert isinstance(analysis.news_sentiment, SentimentScore)
        assert isinstance(analysis.social_sentiment, SentimentScore)
        assert isinstance(analysis.analyst_sentiment, SentimentScore)
        assert analysis.sentiment_trend in ["IMPROVING", "DECLINING", "STABLE"]
        assert isinstance(analysis.key_themes, list)
        assert isinstance(analysis.risk_factors, list)
    
    def test_analyze_complete_negative(self):
        """Test complete sentiment analysis with negative data."""
        analysis = self.analyzer.analyze(
            symbol="AAPL",
            news_data=[self.negative_news],
            social_data=self.negative_social,
            analyst_reports=["Downgrade to sell with concerns about future growth"],
            historical_sentiment=[0.3, 0.1, -0.1]
        )
        
        assert analysis.overall_sentiment == SentimentPolarity.NEGATIVE
        assert analysis.overall_score < 0
        assert analysis.sentiment_trend == "DECLINING"
    
    def test_analyze_complete_mixed(self):
        """Test complete sentiment analysis with mixed data."""
        analysis = self.analyzer.analyze(
            symbol="AAPL",
            news_data=[self.positive_news, self.negative_news],
            social_data=self.positive_social + self.negative_social,
            analyst_reports=["Hold rating with mixed outlook"],
            historical_sentiment=[0.0, 0.1, 0.0]
        )
        
        # Should handle mixed sentiment appropriately
        assert isinstance(analysis.overall_sentiment, SentimentPolarity)
        assert -1 <= analysis.overall_score <= 1
        assert 0 <= analysis.overall_confidence <= 1
    
    def test_analyze_complete_empty_data(self):
        """Test complete sentiment analysis with empty data."""
        analysis = self.analyzer.analyze(
            symbol="AAPL",
            news_data=[],
            social_data=[],
            analyst_reports=[],
            historical_sentiment=None
        )
        
        assert analysis.overall_sentiment == SentimentPolarity.NEUTRAL
        assert analysis.overall_score == 0.0
        assert analysis.overall_confidence == 0.0
        assert analysis.sentiment_trend == "STABLE"
    
    def test_sentiment_score_to_dict(self):
        """Test SentimentScore to_dict method."""
        score = SentimentScore(
            polarity=SentimentPolarity.POSITIVE,
            confidence=0.8,
            score=0.6,
            keywords=["bullish", "growth", "profit"]
        )
        
        result = score.to_dict()
        
        assert isinstance(result, dict)
        assert result["polarity"] == "POSITIVE"
        assert result["confidence"] == 0.8
        assert result["score"] == 0.6
        assert result["keywords"] == ["bullish", "growth", "profit"]
    
    def test_news_data_to_dict(self):
        """Test NewsData to_dict method."""
        result = self.positive_news.to_dict()
        
        assert isinstance(result, dict)
        assert result["title"] == self.positive_news.title
        assert result["content"] == self.positive_news.content
        assert result["source"] == "FINANCIAL_NEWS"
        assert result["symbol"] == "AAPL"
        assert "timestamp" in result
    
    def test_sentiment_analysis_to_dict(self):
        """Test SentimentAnalysis to_dict method."""
        analysis = self.analyzer.analyze(
            symbol="AAPL",
            news_data=[self.positive_news],
            social_data=self.positive_social,
            analyst_reports=["Positive outlook"]
        )
        
        result = analysis.to_dict()
        
        assert isinstance(result, dict)
        assert result["symbol"] == "AAPL"
        assert "timestamp" in result
        assert result["overall_sentiment"] in ["POSITIVE", "NEGATIVE", "NEUTRAL"]
        assert "overall_score" in result
        assert "overall_confidence" in result
        assert "news_sentiment" in result
        assert "social_sentiment" in result
        assert "analyst_sentiment" in result
        assert "sentiment_trend" in result
        assert "key_themes" in result
        assert "risk_factors" in result


class TestSentimentAnalysisAccuracy:
    """Test accuracy of sentiment analysis with known examples."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
    
    def test_strong_positive_sentiment(self):
        """Test recognition of strongly positive sentiment."""
        text = "record breaking earnings beat expectations surge in revenue bullish outlook strong growth momentum"
        score = self.analyzer.calculate_sentiment_score(text)
        
        assert score.polarity == SentimentPolarity.POSITIVE
        assert score.score > 0.5
        assert score.confidence > 0.5
    
    def test_strong_negative_sentiment(self):
        """Test recognition of strongly negative sentiment."""
        text = "massive losses plunge in stock price bearish outlook weak performance declining revenue"
        score = self.analyzer.calculate_sentiment_score(text)
        
        assert score.polarity == SentimentPolarity.NEGATIVE
        assert score.score <= -0.5
        assert score.confidence > 0.5
    
    def test_mixed_sentiment_with_risk(self):
        """Test handling of mixed sentiment with risk factors."""
        text = "strong earnings growth but high volatility and regulatory uncertainty create risk"
        score = self.analyzer.calculate_sentiment_score(text)
        
        # Risk should temper the positive sentiment
        assert score.score < 0.5  # Less positive due to risk
        assert "risk" in score.keywords or "volatility" in score.keywords
    
    def test_financial_context_awareness(self):
        """Test that analyzer understands financial context."""
        financial_text = "eps beat guidance revenue growth margin expansion"
        general_text = "good nice excellent wonderful"
        
        financial_score = self.analyzer.calculate_sentiment_score(financial_text)
        general_score = self.analyzer.calculate_sentiment_score(general_text)
        
        # Financial text should have higher confidence due to domain-specific keywords
        assert financial_score.confidence >= general_score.confidence
    
    def test_keyword_density_impact(self):
        """Test that keyword density affects confidence."""
        sparse_text = "good earnings in a very long report with many other unrelated words and topics that dilute the sentiment signal significantly"
        dense_text = "excellent earnings strong growth bullish outlook"
        
        sparse_score = self.analyzer.calculate_sentiment_score(sparse_text)
        dense_score = self.analyzer.calculate_sentiment_score(dense_text)
        
        # Dense text should have higher or equal confidence
        assert dense_score.confidence >= sparse_score.confidence


if __name__ == "__main__":
    pytest.main([__file__])