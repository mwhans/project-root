export interface MarketMakingConfig {
    // Spread and Pricing
    baseSpreadPercentage: number;    // Base spread around mid price
    dynamicSpreadMultiplier: number; // Multiplier for volatility-based spread adjustment
    minSpreadPercentage: number;     // Minimum spread to maintain
    maxSpreadPercentage: number;     // Maximum spread to maintain
    
    // Position Management
    maxPositionSize: number;         // Maximum position size in base currency
    minOrderSize: number;            // Minimum order size
    maxOrderSize: number;            // Maximum order size per order
    
    // Risk Management
    maxDailyLoss: number;           // Maximum allowed daily loss
    maxDrawdown: number;            // Maximum allowed drawdown
    minLiquidityUsd: number;        // Minimum required liquidity
    
    // Order Management
    orderRefreshTime: number;        // Time in ms before refreshing orders
    maxOpenOrders: number;          // Maximum number of open orders per side
    
    // Market Conditions
    maxVolatility: number;          // Maximum allowed volatility
    minVolumeUsd: number;           // Minimum required 24h volume
} 