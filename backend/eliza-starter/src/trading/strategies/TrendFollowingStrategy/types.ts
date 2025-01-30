export interface TechnicalIndicators {
    sma: {
        fast: number;
        slow: number;
    };
    ema: {
        fast: number;
        slow: number;
    };
    macd: {
        value: number;
        signal: number;
        histogram: number;
    };
    rsi: number;
    atr: number;
    volume: {
        current: number;
        average: number;
    };
}

export interface TrendState {
    direction: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
    strength: number;  // 0-1
    duration: number;  // in periods
    confirmation: number;  // 0-1, based on indicator agreement
}

export interface TrendFollowingConfig {
    // Indicator Periods
    fastMaPeriod: number;
    slowMaPeriod: number;
    rsiPeriod: number;
    macdConfig: {
        fastPeriod: number;
        slowPeriod: number;
        signalPeriod: number;
    };
    
    // Entry/Exit Conditions
    rsiOverbought: number;
    rsiOversold: number;
    minTrendStrength: number;
    minConfirmation: number;
    
    // Position Management
    maxPositionSize: number;
    positionSizeAdjustment: number;  // 0-1
    stopLossPercentage: number;
    takeProfitPercentage: number;
    
    // Risk Management
    maxDrawdown: number;
    trailingStopDistance: number;
} 