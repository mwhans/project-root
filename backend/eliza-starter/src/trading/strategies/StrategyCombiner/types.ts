export type CombinationMethod = 'MAJORITY_VOTE' | 'WEIGHTED_VOTE' | 
    'CONSENSUS' | 'PRIORITY' | 'SEQUENTIAL';

export type ConflictResolution = 'IGNORE' | 'HOLD' | 'WEIGHTED_AVERAGE' | 
    'STRONGEST_SIGNAL' | 'MOST_CONFIDENT';

export interface StrategyWeight {
    strategyName: string;
    weight: number;
    priority?: number;
    minConfidence?: number;
}

export interface CombinerConfig {
    // Signal Combination
    combinationMethod: CombinationMethod;
    conflictResolution: ConflictResolution;
    strategyWeights: StrategyWeight[];
    
    // Thresholds
    minStrategyAgreement: number;     // % of strategies that must agree
    minConfidenceThreshold: number;
    minSignalStrength: number;
    
    // Risk Management
    maxPositionSize: number;
    maxSimultaneousSignals: number;
    riskPerTrade: number;
    
    // Performance Tracking
    performanceWindow: number;        // ms to track strategy performance
    adaptiveWeights: boolean;         // dynamically adjust weights
    
    // Execution
    sequentialExecution: boolean;     // execute strategies in sequence
    executionTimeout: number;         // ms to wait for execution
    retryAttempts: number;
}

export interface StrategyPerformance {
    strategyName: string;
    signals: number;
    successRate: number;
    profitFactor: number;
    sharpeRatio: number;
    weight: number;
    lastUpdated: number;
}

export interface CombinedSignal {
    original: StrategySignal[];
    combined: StrategySignal;
    confidence: number;
    agreement: number;
    strategies: string[];
} 