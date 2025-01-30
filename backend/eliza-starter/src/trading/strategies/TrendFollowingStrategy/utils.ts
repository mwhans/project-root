import { TechnicalIndicators, TrendState } from './types';

export function calculateSMA(prices: number[], period: number): number {
    if (prices.length < period) return 0;
    const slice = prices.slice(-period);
    return slice.reduce((sum, price) => sum + price, 0) / period;
}

export function calculateEMA(prices: number[], period: number, prevEMA?: number): number {
    if (prices.length < period) return 0;
    if (!prevEMA) {
        return calculateSMA(prices.slice(-period), period);
    }
    
    const multiplier = 2 / (period + 1);
    const currentPrice = prices[prices.length - 1];
    return (currentPrice - prevEMA) * multiplier + prevEMA;
}

export function calculateMACD(
    prices: number[],
    fastPeriod: number,
    slowPeriod: number,
    signalPeriod: number
): { value: number; signal: number; histogram: number } {
    const fastEMA = calculateEMA(prices, fastPeriod);
    const slowEMA = calculateEMA(prices, slowPeriod);
    const macdValue = fastEMA - slowEMA;
    
    // Calculate signal line (EMA of MACD)
    const macdHistory = prices.map((_, i) => {
        const slice = prices.slice(0, i + 1);
        return calculateEMA(slice, fastPeriod) - calculateEMA(slice, slowPeriod);
    });
    
    const signal = calculateEMA(macdHistory, signalPeriod);
    const histogram = macdValue - signal;

    return { value: macdValue, signal, histogram };
}

export function calculateRSI(prices: number[], period: number): number {
    if (prices.length < period + 1) return 50;

    let gains = 0;
    let losses = 0;

    for (let i = 1; i <= period; i++) {
        const difference = prices[prices.length - i] - prices[prices.length - i - 1];
        if (difference >= 0) {
            gains += difference;
        } else {
            losses -= difference;
        }
    }

    const avgGain = gains / period;
    const avgLoss = losses / period;
    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
}

export function calculateATR(
    highs: number[],
    lows: number[],
    closes: number[],
    period: number
): number {
    if (highs.length < period) return 0;

    const trs = highs.map((high, i) => {
        if (i === 0) return high - lows[i];
        const prevClose = closes[i - 1];
        return Math.max(
            high - lows[i],
            Math.abs(high - prevClose),
            Math.abs(lows[i] - prevClose)
        );
    });

    return calculateEMA(trs, period);
}

export function analyzeTrendState(
    indicators: TechnicalIndicators,
    config: any
): TrendState {
    // Determine trend direction
    const macdTrend = indicators.macd.histogram > 0 ? 'BULLISH' : 'BEARISH';
    const emaTrend = indicators.ema.fast > indicators.ema.slow ? 'BULLISH' : 'BEARISH';
    const rsiTrend = indicators.rsi > 50 ? 'BULLISH' : 'BEARISH';

    // Calculate trend strength
    const trendStrength = calculateTrendStrength(indicators);

    // Calculate confirmation level
    const confirmation = [
        macdTrend === 'BULLISH',
        emaTrend === 'BULLISH',
        rsiTrend === 'BULLISH',
        indicators.volume.current > indicators.volume.average
    ].filter(Boolean).length / 4;

    return {
        direction: confirmation > 0.5 ? 'BULLISH' : 'BEARISH',
        strength: trendStrength,
        duration: 0, // This should be tracked in the strategy state
        confirmation
    };
}

function calculateTrendStrength(indicators: TechnicalIndicators): number {
    const macdStrength = Math.abs(indicators.macd.histogram) / Math.abs(indicators.macd.value);
    const emaStrength = Math.abs(indicators.ema.fast - indicators.ema.slow) / indicators.ema.slow;
    const rsiStrength = Math.abs(indicators.rsi - 50) / 50;
    const volumeStrength = indicators.volume.current / indicators.volume.average;

    return (macdStrength + emaStrength + rsiStrength + volumeStrength) / 4;
} 