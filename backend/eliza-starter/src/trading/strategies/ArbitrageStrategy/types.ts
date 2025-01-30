export interface ExchangeQuote {
    exchange: string;
    price: number;
    liquidityUsd: number;
    fees: {
        maker: number;
        taker: number;
    };
}

export interface TokenPair {
    baseToken: string;
    quoteToken: string;
}

export interface ArbitrageOpportunity {
    type: 'SIMPLE' | 'TRIANGULAR';
    path: {
        exchange: string;
        token: string;
        action: 'BUY' | 'SELL';
        price: number;
        amount: number;
    }[];
    expectedProfitUsd: number;
    confidence: number;
}

export interface ArbitrageConfig {
    // Profit thresholds
    minProfitUsd: number;
    minProfitPercentage: number;
    
    // Risk Management
    maxTradeSize: number;
    maxExposureUsd: number;
    minLiquidityUsd: number;
    
    // Execution
    maxExecutionTime: number;  // ms
    maxSlippage: number;
    
    // Monitoring
    profitMonitoringInterval: number;
    maxConsecutiveLosses: number;
    
    // Exchange specific
    enabledExchanges: string[];
    exchangePriority: string[];
} 