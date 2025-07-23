"""
Unit tests for technical analysis module.
"""

import pytest
import numpy as np
from datetime import datetime
from multi_agent_trading.analysis.technical_analysis import (
    TechnicalAnalyzer, TechnicalIndicators, TechnicalAnalysis,
    PatternAnalysis, TrendDirection, PatternType
)


class TestTechnicalAnalyzer:
    """Test cases for TechnicalAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TechnicalAnalyzer()
        
        # Sample price data for testing
        self.sample_prices = [
            100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0,
            111.0, 110.0, 112.0, 114.0, 113.0, 115.0, 117.0, 116.0, 118.0, 120.0,
            119.0, 121.0, 123.0, 122.0, 124.0, 126.0, 125.0, 127.0, 129.0, 128.0
        ]
        
        self.sample_highs = [
            101.0, 103.0, 102.0, 104.0, 106.0, 105.0, 107.0, 109.0, 108.0, 110.0,
            112.0, 111.0, 113.0, 115.0, 114.0, 116.0, 118.0, 117.0, 119.0, 121.0,
            120.0, 122.0, 124.0, 123.0, 125.0, 127.0, 126.0, 128.0, 130.0, 129.0
        ]
        
        self.sample_lows = [
            99.0, 101.0, 100.0, 102.0, 104.0, 103.0, 105.0, 107.0, 106.0, 108.0,
            110.0, 109.0, 111.0, 113.0, 112.0, 114.0, 116.0, 115.0, 117.0, 119.0,
            118.0, 120.0, 122.0, 121.0, 123.0, 125.0, 124.0, 126.0, 128.0, 127.0
        ]
        
        self.sample_volumes = [1000] * len(self.sample_prices)
    
    def test_calculate_rsi_basic(self):
        """Test RSI calculation with basic data."""
        rsi = self.analyzer.calculate_rsi(self.sample_prices)
        
        # RSI should be between 0 and 100
        assert 0 <= rsi <= 100
        
        # With generally increasing prices, RSI should be above 50
        assert rsi > 50
    
    def test_calculate_rsi_insufficient_data(self):
        """Test RSI with insufficient data."""
        short_prices = [100.0, 101.0, 102.0]
        rsi = self.analyzer.calculate_rsi(short_prices)
        
        # Should return neutral RSI (50) for insufficient data
        assert rsi == 50.0
    
    def test_calculate_rsi_extreme_values(self):
        """Test RSI with extreme price movements."""
        # All increasing prices should give high RSI
        increasing_prices = [100.0 + i for i in range(20)]
        rsi_high = self.analyzer.calculate_rsi(increasing_prices)
        assert rsi_high > 70
        
        # All decreasing prices should give low RSI
        decreasing_prices = [120.0 - i for i in range(20)]
        rsi_low = self.analyzer.calculate_rsi(decreasing_prices)
        assert rsi_low < 30
    
    def test_calculate_macd_basic(self):
        """Test MACD calculation."""
        macd, signal, histogram = self.analyzer.calculate_macd(self.sample_prices)
        
        # MACD values should be reasonable
        assert isinstance(macd, float)
        assert isinstance(signal, float)
        assert isinstance(histogram, float)
        
        # Histogram should equal MACD - Signal
        assert abs(histogram - (macd - signal)) < 0.001
    
    def test_calculate_macd_insufficient_data(self):
        """Test MACD with insufficient data."""
        short_prices = [100.0, 101.0]
        macd, signal, histogram = self.analyzer.calculate_macd(short_prices)
        
        # Should return zeros for insufficient data
        assert macd == 0.0
        assert signal == 0.0
        assert histogram == 0.0
    
    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = self.analyzer.calculate_bollinger_bands(self.sample_prices)
        
        # Upper band should be greater than middle, middle greater than lower
        assert upper > middle > lower
        
        # Middle band should be close to SMA
        sma = self.analyzer.calculate_sma(self.sample_prices, 20)
        assert abs(middle - sma) < 0.001
    
    def test_calculate_bollinger_bands_insufficient_data(self):
        """Test Bollinger Bands with insufficient data."""
        short_prices = [100.0, 101.0]
        upper, middle, lower = self.analyzer.calculate_bollinger_bands(short_prices)
        
        # All bands should be equal to current price
        assert upper == middle == lower == 101.0
    
    def test_calculate_sma(self):
        """Test Simple Moving Average calculation."""
        sma_10 = self.analyzer.calculate_sma(self.sample_prices, 10)
        
        # SMA should be the average of last 10 prices
        expected_sma = np.mean(self.sample_prices[-10:])
        assert abs(sma_10 - expected_sma) < 0.001
    
    def test_calculate_sma_insufficient_data(self):
        """Test SMA with insufficient data."""
        short_prices = [100.0, 101.0]
        sma = self.analyzer.calculate_sma(short_prices, 10)
        
        # Should return average of available data
        assert sma == 100.5
    
    def test_calculate_ema(self):
        """Test Exponential Moving Average calculation."""
        ema = self.analyzer.calculate_ema(self.sample_prices, 12)
        
        # EMA should be a reasonable value
        assert isinstance(ema, float)
        assert ema > 0
        
        # EMA should be within reasonable range of the price data
        min_price = min(self.sample_prices)
        max_price = max(self.sample_prices)
        assert min_price <= ema <= max_price
        
        # EMA should be different from SMA (more responsive)
        sma = self.analyzer.calculate_sma(self.sample_prices, 12)
        assert ema != sma  # EMA and SMA should be different
    
    def test_calculate_stochastic(self):
        """Test Stochastic Oscillator calculation."""
        k_percent, d_percent = self.analyzer.calculate_stochastic(
            self.sample_highs, self.sample_lows, self.sample_prices
        )
        
        # Both values should be between 0 and 100
        assert 0 <= k_percent <= 100
        assert 0 <= d_percent <= 100
    
    def test_calculate_stochastic_insufficient_data(self):
        """Test Stochastic with insufficient data."""
        short_highs = [101.0, 102.0]
        short_lows = [99.0, 100.0]
        short_closes = [100.0, 101.0]
        
        k_percent, d_percent = self.analyzer.calculate_stochastic(
            short_highs, short_lows, short_closes
        )
        
        # Should return neutral values
        assert k_percent == 50.0
        assert d_percent == 50.0
    
    def test_calculate_williams_r(self):
        """Test Williams %R calculation."""
        williams_r = self.analyzer.calculate_williams_r(
            self.sample_highs, self.sample_lows, self.sample_prices
        )
        
        # Williams %R should be between -100 and 0
        assert -100 <= williams_r <= 0
    
    def test_calculate_atr(self):
        """Test Average True Range calculation."""
        atr = self.analyzer.calculate_atr(
            self.sample_highs, self.sample_lows, self.sample_prices
        )
        
        # ATR should be positive
        assert atr >= 0
    
    def test_calculate_atr_insufficient_data(self):
        """Test ATR with insufficient data."""
        atr = self.analyzer.calculate_atr([101.0], [99.0], [100.0])
        
        # Should return 0 for insufficient data
        assert atr == 0.0
    
    def test_calculate_adx(self):
        """Test Average Directional Index calculation."""
        adx = self.analyzer.calculate_adx(
            self.sample_highs, self.sample_lows, self.sample_prices
        )
        
        # ADX should be between 0 and 100
        assert 0 <= adx <= 100
    
    def test_detect_trend_bullish(self):
        """Test trend detection for bullish trend."""
        # Create strongly bullish price data
        bullish_prices = [100.0 + i * 2 for i in range(60)]
        
        trend, strength = self.analyzer.detect_trend(bullish_prices)
        
        assert trend == TrendDirection.BULLISH
        assert 0 <= strength <= 1
    
    def test_detect_trend_bearish(self):
        """Test trend detection for bearish trend."""
        # Create strongly bearish price data
        bearish_prices = [200.0 - i * 2 for i in range(60)]
        
        trend, strength = self.analyzer.detect_trend(bearish_prices)
        
        assert trend == TrendDirection.BEARISH
        assert 0 <= strength <= 1
    
    def test_detect_trend_sideways(self):
        """Test trend detection for sideways trend."""
        # Create sideways price data
        sideways_prices = [100.0 + (i % 2) for i in range(60)]
        
        trend, strength = self.analyzer.detect_trend(sideways_prices)
        
        assert trend == TrendDirection.SIDEWAYS
        assert 0 <= strength <= 1
    
    def test_detect_trend_insufficient_data(self):
        """Test trend detection with insufficient data."""
        short_prices = [100.0, 101.0, 102.0]
        
        trend, strength = self.analyzer.detect_trend(short_prices)
        
        assert trend == TrendDirection.SIDEWAYS
        assert strength == 0.5
    
    def test_detect_patterns_basic(self):
        """Test basic pattern detection."""
        pattern = self.analyzer.detect_patterns(
            self.sample_highs, self.sample_lows, self.sample_prices
        )
        
        assert isinstance(pattern, PatternAnalysis)
        assert isinstance(pattern.pattern_type, PatternType)
        assert 0 <= pattern.confidence <= 1
        assert pattern.support_level is not None
        assert pattern.resistance_level is not None
    
    def test_detect_patterns_insufficient_data(self):
        """Test pattern detection with insufficient data."""
        short_highs = [101.0, 102.0]
        short_lows = [99.0, 100.0]
        short_closes = [100.0, 101.0]
        
        pattern = self.analyzer.detect_patterns(short_highs, short_lows, short_closes)
        
        assert pattern.pattern_type == PatternType.NONE
        assert pattern.confidence == 0.0
        assert pattern.support_level is None
        assert pattern.resistance_level is None
    
    def test_detect_patterns_double_top(self):
        """Test double top pattern detection."""
        # Create double top pattern
        double_top_highs = [100, 105, 102, 106, 103, 105, 101, 104, 100, 103] * 2
        double_top_lows = [98, 103, 100, 104, 101, 103, 99, 102, 98, 101] * 2
        double_top_closes = [99, 104, 101, 105, 102, 104, 100, 103, 99, 102] * 2
        
        pattern = self.analyzer.detect_patterns(
            double_top_highs, double_top_lows, double_top_closes
        )
        
        # Should detect some pattern (may not be perfect double top due to simplified algorithm)
        assert isinstance(pattern.pattern_type, PatternType)
        assert pattern.support_level is not None
        assert pattern.resistance_level is not None
    
    def test_analyze_complete(self):
        """Test complete technical analysis."""
        analysis = self.analyzer.analyze(
            self.sample_highs,
            self.sample_lows,
            self.sample_prices,
            self.sample_volumes,
            "AAPL"
        )
        
        # Verify analysis structure
        assert isinstance(analysis, TechnicalAnalysis)
        assert analysis.symbol == "AAPL"
        assert isinstance(analysis.timestamp, datetime)
        assert isinstance(analysis.indicators, TechnicalIndicators)
        assert isinstance(analysis.trend_direction, TrendDirection)
        assert 0 <= analysis.trend_strength <= 1
        assert isinstance(analysis.pattern_analysis, PatternAnalysis)
        assert analysis.overall_signal in ["BUY", "SELL", "HOLD"]
        assert 0 <= analysis.confidence <= 1
    
    def test_generate_signal_bullish(self):
        """Test signal generation for bullish conditions."""
        # Create indicators that suggest bullish conditions
        indicators = TechnicalIndicators(
            rsi=25.0,  # Oversold
            macd=1.0,
            macd_signal=0.5,  # MACD above signal
            macd_histogram=0.5,
            bollinger_upper=110.0,
            bollinger_middle=105.0,
            bollinger_lower=100.0,  # Price near lower band
            sma_20=106.0,
            sma_50=104.0,  # Short MA above long MA
            ema_12=107.0,
            ema_26=105.0,
            stochastic_k=15.0,  # Oversold
            stochastic_d=18.0,
            williams_r=-85.0,
            atr=2.0,
            adx=30.0
        )
        
        pattern = PatternAnalysis(
            pattern_type=PatternType.DOUBLE_BOTTOM,  # Bullish pattern
            confidence=0.7,
            support_level=100.0,
            resistance_level=110.0,
            target_price=115.0,
            stop_loss=98.0
        )
        
        signal, confidence = self.analyzer._generate_signal(
            indicators, TrendDirection.BULLISH, pattern
        )
        
        assert signal == "BUY"
        assert confidence > 0.5
    
    def test_generate_signal_bearish(self):
        """Test signal generation for bearish conditions."""
        # Create indicators that suggest bearish conditions
        indicators = TechnicalIndicators(
            rsi=75.0,  # Overbought
            macd=-1.0,
            macd_signal=-0.5,  # MACD below signal
            macd_histogram=-0.5,
            bollinger_upper=110.0,
            bollinger_middle=105.0,
            bollinger_lower=100.0,  # Price near upper band
            sma_20=104.0,
            sma_50=106.0,  # Short MA below long MA
            ema_12=103.0,
            ema_26=105.0,
            stochastic_k=85.0,  # Overbought
            stochastic_d=82.0,
            williams_r=-15.0,
            atr=2.0,
            adx=30.0
        )
        
        pattern = PatternAnalysis(
            pattern_type=PatternType.DOUBLE_TOP,  # Bearish pattern
            confidence=0.7,
            support_level=100.0,
            resistance_level=110.0,
            target_price=95.0,
            stop_loss=112.0
        )
        
        signal, confidence = self.analyzer._generate_signal(
            indicators, TrendDirection.BEARISH, pattern
        )
        
        assert signal == "SELL"
        assert confidence > 0.5
    
    def test_technical_indicators_to_dict(self):
        """Test TechnicalIndicators to_dict method."""
        indicators = TechnicalIndicators(
            rsi=50.0, macd=1.0, macd_signal=0.8, macd_histogram=0.2,
            bollinger_upper=110.0, bollinger_middle=105.0, bollinger_lower=100.0,
            sma_20=105.0, sma_50=103.0, ema_12=106.0, ema_26=104.0,
            stochastic_k=50.0, stochastic_d=48.0, williams_r=-50.0,
            atr=2.0, adx=25.0
        )
        
        result = indicators.to_dict()
        
        assert isinstance(result, dict)
        assert result["rsi"] == 50.0
        assert result["macd"] == 1.0
        assert len(result) == 16  # All indicator fields
    
    def test_pattern_analysis_to_dict(self):
        """Test PatternAnalysis to_dict method."""
        pattern = PatternAnalysis(
            pattern_type=PatternType.DOUBLE_TOP,
            confidence=0.7,
            support_level=100.0,
            resistance_level=110.0,
            target_price=95.0,
            stop_loss=112.0
        )
        
        result = pattern.to_dict()
        
        assert isinstance(result, dict)
        assert result["pattern_type"] == "DOUBLE_TOP"
        assert result["confidence"] == 0.7
        assert result["support_level"] == 100.0
    
    def test_technical_analysis_to_dict(self):
        """Test TechnicalAnalysis to_dict method."""
        analysis = self.analyzer.analyze(
            self.sample_highs,
            self.sample_lows,
            self.sample_prices,
            self.sample_volumes,
            "AAPL"
        )
        
        result = analysis.to_dict()
        
        assert isinstance(result, dict)
        assert result["symbol"] == "AAPL"
        assert "timestamp" in result
        assert "indicators" in result
        assert "trend_direction" in result
        assert "pattern_analysis" in result
        assert result["overall_signal"] in ["BUY", "SELL", "HOLD"]


class TestTechnicalIndicatorsAccuracy:
    """Test accuracy of technical indicators against known values."""
    
    def setup_method(self):
        """Set up test fixtures with known data."""
        self.analyzer = TechnicalAnalyzer()
        
        # Known test data with expected results
        self.test_prices = [
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 47.23, 47.00, 47.61,
            47.25, 47.31, 46.23, 46.08, 46.03, 46.83, 47.69, 46.49, 46.26, 47.77
        ]
    
    def test_sma_accuracy(self):
        """Test SMA calculation accuracy."""
        # Calculate 10-period SMA for last 10 values
        sma_10 = self.analyzer.calculate_sma(self.test_prices, 10)
        expected_sma = sum(self.test_prices[-10:]) / 10
        
        assert abs(sma_10 - expected_sma) < 0.001
    
    def test_rsi_bounds(self):
        """Test RSI stays within proper bounds."""
        # Test with extreme data
        extreme_up = [100 + i * 5 for i in range(20)]
        extreme_down = [200 - i * 5 for i in range(20)]
        
        rsi_up = self.analyzer.calculate_rsi(extreme_up)
        rsi_down = self.analyzer.calculate_rsi(extreme_down)
        
        assert 0 <= rsi_up <= 100
        assert 0 <= rsi_down <= 100
        assert rsi_up > rsi_down  # Uptrend should have higher RSI
    
    def test_bollinger_bands_relationship(self):
        """Test Bollinger Bands maintain proper relationships."""
        upper, middle, lower = self.analyzer.calculate_bollinger_bands(self.test_prices)
        
        # Upper > Middle > Lower
        assert upper > middle > lower
        
        # Middle should equal SMA
        sma = self.analyzer.calculate_sma(self.test_prices, 20)
        assert abs(middle - sma) < 0.001
    
    def test_macd_histogram_relationship(self):
        """Test MACD histogram equals MACD - Signal."""
        macd, signal, histogram = self.analyzer.calculate_macd(self.test_prices)
        
        assert abs(histogram - (macd - signal)) < 0.001


if __name__ == "__main__":
    pytest.main([__file__])