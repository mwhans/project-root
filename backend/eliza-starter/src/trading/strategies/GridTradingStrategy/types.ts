export interface GridLevel {
    price: number;
    side: 'BUY' | 'SELL';
    size: number;
    orderId?: string;
    status: 'PENDING' | 'ACTIVE' | 'FILLED' | 'CANCELLED';
    timestamp: number;
}

export interface GridConfig {
    // Grid Parameters
    gridLevels: number;
    gridSpacing: number;        // Percentage between levels
    dynamicSpacing: boolean;    // Use volatility-based spacing
    
    // Position Management
    baseOrderSize: number;      // Base size for each grid level
    sizeMultiplier: number;     // Multiply size as grid levels increase
    maxTotalPositionSize: number;
    
    // Grid Boundaries
    upperBoundary?: number;     // Optional upper price limit
    lowerBoundary?: number;     // Optional lower price limit
    
    // Risk Management
    maxLevelsPerSide: number;
    stopLossLevel: number;      // Price level for emergency stop
    maxDrawdown: number;
    
    // Grid Adjustment
    rebalanceThreshold: number; // % price move to trigger rebalance
    minProfitPerTrade: number;  // Minimum profit to take per grid level
    
    // Execution
    orderTimeout: number;       // ms to wait for order placement
    rebalanceInterval: number;  // ms between grid rebalances
}

export interface GridState {
    referencePrice: number;
    levels: GridLevel[];
    totalInvestment: number;
    currentProfit: number;
    lastRebalance: number;
    activeOrders: Map<string, GridLevel>;
} 