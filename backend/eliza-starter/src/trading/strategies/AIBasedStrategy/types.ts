export type ModelType = 'LSTM' | 'RandomForest' | 'NeuralNetwork' | 'Ensemble';

export interface FeatureSet {
    price: number[];
    volume: number[];
    technicalIndicators: {
        rsi: number[];
        macd: number[];
        bbands: {
            upper: number[];
            middle: number[];
            lower: number[];
        };
    };
    marketMetrics: {
        volatility: number;
        liquidity: number;
        spreadPercentage: number;
    };
    timestamp: number;
}

export interface ModelPrediction {
    direction: 'UP' | 'DOWN' | 'NEUTRAL';
    probability: number;
    confidence: number;
    horizon: number;  // prediction timeframe in minutes
    features: string[];  // features used for prediction
}

export interface AIModelConfig {
    // Model Configuration
    modelType: ModelType;
    modelPath: string;
    updateInterval: number;  // ms between model updates
    
    // Feature Engineering
    featureLookback: number;
    technicalFeatures: string[];
    marketFeatures: string[];
    
    // Prediction Parameters
    confidenceThreshold: number;
    minProbability: number;
    ensembleWeights?: Record<ModelType, number>;
    
    // Position Management
    maxPositionSize: number;
    positionSizeScaler: number;
    
    // Risk Management
    stopLossPercent: number;
    takeProfitPercent: number;
    maxDrawdown: number;
    
    // Model Performance
    performanceMetrics: {
        minAccuracy: number;
        minPrecision: number;
        maxDrawdown: number;
    };
}

export interface ModelMetrics {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
    profitFactor: number;
    sharpeRatio: number;
    drawdown: number;
} 