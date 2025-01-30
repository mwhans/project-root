import { TradeManager } from '../TradeManager';

export interface MarketData {
    price: number;
    volume24h: number;
    liquidityUsd: number;
    priceChange24h: number;
    timestamp: number;
    indicators?: {
        rsi?: number;
        macd?: {
            value: number;
            signal: number;
            histogram: number;
        };
        ema?: {
            short: number;
            long: number;
        };
    };
    orderBook?: OrderBookData;
    averageVolume24h?: number;
    volatility24h?: number;  // 24h price volatility
}

export interface StrategySignal {
    action: "BUY" | "SELL" | "HOLD";
    tokenAddress: string;
    amount: number;
    confidence?: number;
    notes?: string;
    metadata?: Record<string, any>;
    price?: number;        // Limit price for the order
    orderType?: 'MARKET' | 'LIMIT';
}

export interface ITradingStrategy {
    strategyName: string;
    analyzeMarket(marketData: MarketData): Promise<StrategySignal[]>;
    executeTrades(signals: StrategySignal[], tradeManager: TradeManager): Promise<void>;
    updateState(updateParams: any): void;
}

export interface OrderBookData {
    bids: Array<[number, number]>; // price, size pairs
    asks: Array<[number, number]>;
    bestBid: number;
    bestAsk: number;
    spread: number;
    timestamp: number;
}

export interface LimitOrderParams {
    tokenAddress: string;
    side: 'BUY' | 'SELL';
    price: number;
    amount: number;
    expiryTime?: number;
}

