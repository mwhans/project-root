export interface PriceStatistics {
    mean: number;
    standardDeviation: number;
    zScore: number;
    bollingerBands: {
        upper: number;
        middle: number;
        lower: number;
    };
    rsi: number;
    volatility: number;
}

export interface MeanReversionConfig {
    // Statistical Parameters
    lookbackPeriod: number;
    bollingerBandPeriod: number;
    bollingerBandStdDev: number;
    rsiPeriod: number;
    
    // Entry/Exit Thresholds
    zScoreThreshold: number;
    rsiOverbought: number;
    rsiOversold: number;
    
    // Position Management
    maxPositionSize: number;
    positionSizeScaling: number;  // 0-1
    stopLossDeviation: number;    // in standard deviations
    takeProfitDeviation: number;  // in standard deviations
    
    // Risk Management
    maxDrawdown: number;
    maxPositions: number;
    minVolume: number;
    maxSpread: number;           // maximum allowed bid-ask spread
    
    // Trade Management
    entryTimeout: number;        // ms to wait for entry
    exitTimeout: number;         // ms to wait for exit
    minProfitTarget: number;     // minimum profit to take
}

export interface RevertSignal {
    strength: number;            // 0-1 indicating reversion probability
    direction: 'UP' | 'DOWN';
    confidence: number;          // 0-1 based on multiple indicators
    expectedReturn: number;      // expected return based on historical data
} 