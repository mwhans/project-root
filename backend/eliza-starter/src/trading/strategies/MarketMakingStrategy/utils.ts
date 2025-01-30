import { MarketData } from '../types';

export function calculateVolatility(marketData: MarketData): number {
    // Simple volatility calculation based on 24h price change
    // In production, you might want to use more sophisticated methods
    return Math.abs(marketData.priceChange24h / 100);
}

export function calculateOptimalSpread(
    baseSpread: number,
    volatility: number,
    multiplier: number,
    minSpread: number,
    maxSpread: number
): number {
    // Adjust spread based on volatility
    const dynamicSpread = baseSpread * (1 + volatility * multiplier);
    return Math.min(Math.max(dynamicSpread, minSpread), maxSpread);
} 