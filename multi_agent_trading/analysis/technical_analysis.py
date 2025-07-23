"""
Technical analysis module for calculating trading indicators and patterns.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum


class TrendDirection(Enum):
    """Trend direction enumeration."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"


class PatternType(Enum):
    """Chart pattern types."""
    DOUBLE_TOP = "DOUBLE_TOP"
    DOUBLE_BOTTOM = "DOUBLE_BOTTOM"
    HEAD_AND_SHOULDERS = "HEAD_AND_SHOULDERS"
    INVERSE_HEAD_AND_SHOULDERS = "INVERSE_HEAD_AND_SHOULDERS"
    TRIANGLE = "TRIANGLE"
    WEDGE = "WEDGE"
    FLAG = "FLAG"
    PENNANT = "PENNANT"
    NONE = "NONE"


@dataclass
class TechnicalIndicators:
    """Container for technical indicator values."""
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bollinger_upper: float
    bollinger_middle: float
    bollinger_lower: float
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    stochastic_k: float
    stochastic_d: float
    williams_r: float
    atr: float
    adx: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "rsi": self.rsi,
            "macd": self.macd,
            "macd_signal": self.macd_signal,
            "macd_histogram": self.macd_histogram,
            "bollinger_upper": self.bollinger_upper,
            "bollinger_middle": self.bollinger_middle,
            "bollinger_lower": self.bollinger_lower,
            "sma_20": self.sma_20,
            "sma_50": self.sma_50,
            "ema_12": self.ema_12,
            "ema_26": self.ema_26,
            "stochastic_k": self.stochastic_k,
            "stochastic_d": self.stochastic_d,
            "williams_r": self.williams_r,
            "atr": self.atr,
            "adx": self.adx
        }


@dataclass
class PatternAnalysis:
    """Pattern recognition analysis result."""
    pattern_type: PatternType
    confidence: float
    support_level: Optional[float]
    resistance_level: Optional[float]
    target_price: Optional[float]
    stop_loss: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_type": self.pattern_type.value,
            "confidence": self.confidence,
            "support_level": self.support_level,
            "resistance_level": self.resistance_level,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss
        }


@dataclass
class TechnicalAnalysis:
    """Complete technical analysis result."""
    symbol: str
    timestamp: datetime
    indicators: TechnicalIndicators
    trend_direction: TrendDirection
    trend_strength: float
    pattern_analysis: PatternAnalysis
    overall_signal: str  # BUY, SELL, HOLD
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "indicators": self.indicators.to_dict(),
            "trend_direction": self.trend_direction.value,
            "trend_strength": self.trend_strength,
            "pattern_analysis": self.pattern_analysis.to_dict(),
            "overall_signal": self.overall_signal,
            "confidence": self.confidence
        }


class TechnicalAnalyzer:
    """Technical analysis calculator for trading indicators."""
    
    def __init__(self):
        """Initialize the technical analyzer."""
        pass
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: List of closing prices
            period: RSI period (default 14)
            
        Returns:
            RSI value (0-100)
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI if insufficient data
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: List of closing prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        if len(prices) < slow:
            return 0.0, 0.0, 0.0
        
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD line)
        macd_values = []
        for i in range(slow - 1, len(prices)):
            ema_fast_val = self._calculate_ema(prices[:i+1], fast)
            ema_slow_val = self._calculate_ema(prices[:i+1], slow)
            macd_values.append(ema_fast_val - ema_slow_val)
        
        if len(macd_values) >= signal:
            signal_line = self._calculate_ema(macd_values, signal)
        else:
            signal_line = macd_line
        
        histogram = macd_line - signal_line
        
        return float(macd_line), float(signal_line), float(histogram)
    
    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: List of closing prices
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        if len(prices) < period:
            current_price = prices[-1] if prices else 0.0
            return current_price, current_price, current_price
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return float(upper_band), float(sma), float(lower_band)
    
    def calculate_sma(self, prices: List[float], period: int) -> float:
        """
        Calculate Simple Moving Average.
        
        Args:
            prices: List of closing prices
            period: Moving average period
            
        Returns:
            SMA value
        """
        if len(prices) < period:
            return np.mean(prices) if prices else 0.0
        
        return float(np.mean(prices[-period:]))
    
    def calculate_ema(self, prices: List[float], period: int) -> float:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: List of closing prices
            period: EMA period
            
        Returns:
            EMA value
        """
        return self._calculate_ema(prices, period)
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Internal EMA calculation."""
        if len(prices) < period:
            return np.mean(prices) if prices else 0.0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return float(ema)
    
    def calculate_stochastic(self, highs: List[float], lows: List[float], closes: List[float], 
                           k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            k_period: %K period
            d_period: %D period
            
        Returns:
            Tuple of (%K, %D)
        """
        if len(closes) < k_period:
            return 50.0, 50.0
        
        highest_high = max(highs[-k_period:])
        lowest_low = min(lows[-k_period:])
        current_close = closes[-1]
        
        if highest_high == lowest_low:
            k_percent = 50.0
        else:
            k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Calculate %D as SMA of %K values
        k_values = []
        for i in range(max(0, len(closes) - d_period), len(closes)):
            if i >= k_period - 1:
                period_high = max(highs[i-k_period+1:i+1])
                period_low = min(lows[i-k_period+1:i+1])
                if period_high != period_low:
                    k_val = ((closes[i] - period_low) / (period_high - period_low)) * 100
                else:
                    k_val = 50.0
                k_values.append(k_val)
        
        d_percent = np.mean(k_values) if k_values else k_percent
        
        return float(k_percent), float(d_percent)
    
    def calculate_williams_r(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """
        Calculate Williams %R.
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            period: Calculation period
            
        Returns:
            Williams %R value
        """
        if len(closes) < period:
            return -50.0
        
        highest_high = max(highs[-period:])
        lowest_low = min(lows[-period:])
        current_close = closes[-1]
        
        if highest_high == lowest_low:
            return -50.0
        
        williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
        
        return float(williams_r)
    
    def calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """
        Calculate Average True Range (ATR).
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            period: ATR period
            
        Returns:
            ATR value
        """
        if len(closes) < 2:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close_prev = abs(highs[i] - closes[i-1])
            low_close_prev = abs(lows[i] - closes[i-1])
            
            true_range = max(high_low, high_close_prev, low_close_prev)
            true_ranges.append(true_range)
        
        if len(true_ranges) < period:
            return np.mean(true_ranges) if true_ranges else 0.0
        
        return float(np.mean(true_ranges[-period:]))
    
    def calculate_adx(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            period: ADX period
            
        Returns:
            ADX value
        """
        if len(closes) < period + 1:
            return 25.0  # Neutral ADX
        
        # Simplified ADX calculation
        # In a real implementation, this would be more complex
        price_changes = [abs(closes[i] - closes[i-1]) for i in range(1, len(closes))]
        
        if len(price_changes) < period:
            return 25.0
        
        avg_change = np.mean(price_changes[-period:])
        max_change = max(price_changes[-period:])
        
        if max_change == 0:
            return 25.0
        
        # Simplified ADX as percentage of average change to max change
        adx = (avg_change / max_change) * 100
        
        return float(min(100.0, max(0.0, adx)))
    
    def detect_trend(self, prices: List[float], short_period: int = 20, long_period: int = 50) -> Tuple[TrendDirection, float]:
        """
        Detect trend direction and strength.
        
        Args:
            prices: List of closing prices
            short_period: Short-term moving average period
            long_period: Long-term moving average period
            
        Returns:
            Tuple of (trend direction, trend strength 0-1)
        """
        if len(prices) < long_period:
            return TrendDirection.SIDEWAYS, 0.5
        
        short_ma = self.calculate_sma(prices, short_period)
        long_ma = self.calculate_sma(prices, long_period)
        current_price = prices[-1]
        
        # Determine trend direction
        if short_ma > long_ma and current_price > short_ma:
            trend = TrendDirection.BULLISH
        elif short_ma < long_ma and current_price < short_ma:
            trend = TrendDirection.BEARISH
        else:
            trend = TrendDirection.SIDEWAYS
        
        # Calculate trend strength based on price deviation from moving averages
        ma_diff = abs(short_ma - long_ma)
        price_range = max(prices[-long_period:]) - min(prices[-long_period:])
        
        if price_range == 0:
            strength = 0.5
        else:
            strength = min(1.0, ma_diff / price_range)
        
        return trend, float(strength)
    
    def detect_patterns(self, highs: List[float], lows: List[float], closes: List[float]) -> PatternAnalysis:
        """
        Detect chart patterns.
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            
        Returns:
            Pattern analysis result
        """
        if len(closes) < 20:
            return PatternAnalysis(
                pattern_type=PatternType.NONE,
                confidence=0.0,
                support_level=None,
                resistance_level=None,
                target_price=None,
                stop_loss=None
            )
        
        # Simplified pattern detection
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]
        recent_closes = closes[-20:]
        
        # Calculate support and resistance levels
        support_level = min(recent_lows)
        resistance_level = max(recent_highs)
        current_price = closes[-1]
        
        # Simple double top/bottom detection
        max_high = max(recent_highs)
        min_low = min(recent_lows)
        
        # Count peaks and troughs
        peaks = self._find_peaks(recent_highs)
        troughs = self._find_peaks([-x for x in recent_lows])
        
        pattern_type = PatternType.NONE
        confidence = 0.0
        target_price = None
        stop_loss = None
        
        if len(peaks) >= 2 and max(recent_highs) in [recent_highs[i] for i in peaks[-2:]]:
            # Potential double top
            pattern_type = PatternType.DOUBLE_TOP
            confidence = 0.6
            target_price = current_price - (resistance_level - support_level)
            stop_loss = resistance_level
        elif len(troughs) >= 2 and min(recent_lows) in [recent_lows[i] for i in troughs[-2:]]:
            # Potential double bottom
            pattern_type = PatternType.DOUBLE_BOTTOM
            confidence = 0.6
            target_price = current_price + (resistance_level - support_level)
            stop_loss = support_level
        
        return PatternAnalysis(
            pattern_type=pattern_type,
            confidence=confidence,
            support_level=support_level,
            resistance_level=resistance_level,
            target_price=target_price,
            stop_loss=stop_loss
        )
    
    def _find_peaks(self, data: List[float], min_distance: int = 3) -> List[int]:
        """Find peaks in price data."""
        peaks = []
        for i in range(min_distance, len(data) - min_distance):
            is_peak = True
            for j in range(i - min_distance, i + min_distance + 1):
                if j != i and data[j] >= data[i]:
                    is_peak = False
                    break
            if is_peak:
                peaks.append(i)
        return peaks
    
    def analyze(self, highs: List[float], lows: List[float], closes: List[float], 
                volumes: List[int], symbol: str) -> TechnicalAnalysis:
        """
        Perform complete technical analysis.
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            volumes: List of volumes
            symbol: Trading symbol
            
        Returns:
            Complete technical analysis
        """
        # Calculate all indicators
        rsi = self.calculate_rsi(closes)
        macd, macd_signal, macd_histogram = self.calculate_macd(closes)
        bollinger_upper, bollinger_middle, bollinger_lower = self.calculate_bollinger_bands(closes)
        sma_20 = self.calculate_sma(closes, 20)
        sma_50 = self.calculate_sma(closes, 50)
        ema_12 = self.calculate_ema(closes, 12)
        ema_26 = self.calculate_ema(closes, 26)
        stochastic_k, stochastic_d = self.calculate_stochastic(highs, lows, closes)
        williams_r = self.calculate_williams_r(highs, lows, closes)
        atr = self.calculate_atr(highs, lows, closes)
        adx = self.calculate_adx(highs, lows, closes)
        
        indicators = TechnicalIndicators(
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            macd_histogram=macd_histogram,
            bollinger_upper=bollinger_upper,
            bollinger_middle=bollinger_middle,
            bollinger_lower=bollinger_lower,
            sma_20=sma_20,
            sma_50=sma_50,
            ema_12=ema_12,
            ema_26=ema_26,
            stochastic_k=stochastic_k,
            stochastic_d=stochastic_d,
            williams_r=williams_r,
            atr=atr,
            adx=adx
        )
        
        # Detect trend
        trend_direction, trend_strength = self.detect_trend(closes)
        
        # Detect patterns
        pattern_analysis = self.detect_patterns(highs, lows, closes)
        
        # Generate overall signal
        overall_signal, confidence = self._generate_signal(indicators, trend_direction, pattern_analysis)
        
        return TechnicalAnalysis(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            indicators=indicators,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            pattern_analysis=pattern_analysis,
            overall_signal=overall_signal,
            confidence=confidence
        )
    
    def _generate_signal(self, indicators: TechnicalIndicators, trend: TrendDirection, 
                        pattern: PatternAnalysis) -> Tuple[str, float]:
        """Generate overall trading signal based on technical analysis."""
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        
        # RSI signals
        if indicators.rsi < 30:
            bullish_signals += 1
        elif indicators.rsi > 70:
            bearish_signals += 1
        total_signals += 1
        
        # MACD signals
        if indicators.macd > indicators.macd_signal and indicators.macd_histogram > 0:
            bullish_signals += 1
        elif indicators.macd < indicators.macd_signal and indicators.macd_histogram < 0:
            bearish_signals += 1
        total_signals += 1
        
        # Bollinger Bands signals
        if indicators.bollinger_lower > 0:  # Avoid division by zero
            current_position = (indicators.bollinger_middle - indicators.bollinger_lower) / (indicators.bollinger_upper - indicators.bollinger_lower)
            if current_position < 0.2:
                bullish_signals += 1
            elif current_position > 0.8:
                bearish_signals += 1
        total_signals += 1
        
        # Moving average signals
        if indicators.sma_20 > indicators.sma_50:
            bullish_signals += 1
        elif indicators.sma_20 < indicators.sma_50:
            bearish_signals += 1
        total_signals += 1
        
        # Stochastic signals
        if indicators.stochastic_k < 20 and indicators.stochastic_d < 20:
            bullish_signals += 1
        elif indicators.stochastic_k > 80 and indicators.stochastic_d > 80:
            bearish_signals += 1
        total_signals += 1
        
        # Trend signals
        if trend == TrendDirection.BULLISH:
            bullish_signals += 2  # Weight trend more heavily
        elif trend == TrendDirection.BEARISH:
            bearish_signals += 2
        total_signals += 2
        
        # Pattern signals
        if pattern.pattern_type in [PatternType.DOUBLE_BOTTOM, PatternType.INVERSE_HEAD_AND_SHOULDERS]:
            bullish_signals += 1
        elif pattern.pattern_type in [PatternType.DOUBLE_TOP, PatternType.HEAD_AND_SHOULDERS]:
            bearish_signals += 1
        total_signals += 1
        
        # Calculate signal strength
        if total_signals == 0:
            signal = "HOLD"
            confidence = 0.5
        elif bullish_signals > bearish_signals:
            signal = "BUY"
            confidence = bullish_signals / total_signals
        elif bearish_signals > bullish_signals:
            signal = "SELL"
            confidence = bearish_signals / total_signals
        else:
            signal = "HOLD"
            confidence = 0.5
        
        return signal, confidence