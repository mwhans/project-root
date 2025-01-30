export interface PriceLevel {
    price: number;
    strength: number;     // 0-1 indicating level strength
    touches: number;      // Number of times price touched this level
    lastTouch: number;    // Timestamp
    type: 'SUPPORT' | 'RESISTANCE';
}

export interface BreakoutConfig {
    // Price Level Detection
    lookbackPeriods: number;
    minTouchPoints: number;
    levelStrengthThreshold: number;
    
    // Breakout Confirmation
    volumeConfirmation: boolean;
    minVolumeIncrease: number;    // Minimum volume increase for confirmation
    rsiConfirmation: boolean;
    rsiThresholds: {
        overbought: number;
        oversold: number;
    };
    
    // Position Management
    positionSize: number;
    scalingFactor: number;        // Position size multiplier based on conviction
    
    // Risk Management
    stopLossPercent: number;
    takeProfitPercent: number;
    maxPositions: number;
    trailingStop: boolean;
    trailingStopDistance: number;
    
    // Breakout Validation
    minBreakoutPercent: number;   // Minimum price move for valid breakout
    confirmationPeriods: number;   // Periods to confirm breakout
    falseBreakoutGuard: boolean;  // Enable protection against false breakouts
}

export interface BreakoutSignal {
    direction: 'UP' | 'DOWN';
    strength: number;             // 0-1 indicating breakout strength
    level: PriceLevel;
    confirmation: {
        volume: boolean;
        rsi: boolean;
        price: boolean;
    };
    expectedMove: number;         // Expected price move after breakout
} 