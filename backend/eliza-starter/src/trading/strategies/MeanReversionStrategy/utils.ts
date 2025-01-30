import { PriceStatistics, RevertSignal } from './types';

export function calculatePriceStatistics(
    prices: number[],
    period: number
): PriceStatistics {
    const recentPrices = prices.slice(-period);
    const mean = calculateMean(recentPrices);
    const stdDev = calculateStandardDeviation(recentPrices, mean);
    const currentPrice = prices[prices.length - 1];
    const zScore = (currentPrice - mean) / stdDev;

    return {
        mean,
        standardDeviation: stdDev,
        zScore,
        bollingerBands: calculateBollingerBands(prices, period),
        rsi: calculateRSI(prices, period),
        volatility: calculateVolatility(prices, period)
    };
}

export function calculateMean(prices: number[]): number {
    return prices.reduce((sum, price) => sum + price, 0) / prices.length;
}

export function calculateStandardDeviation(
    prices: number[],
    mean: number
): number {
    const squaredDiffs = prices.map(price => Math.pow(price - mean, 2));
    const variance = calculateMean(squaredDiffs);
    return Math.sqrt(variance);
}

export function calculateBollingerBands(
    prices: number[],
    period: number,
    stdDevMultiplier: number = 2
): { upper: number; middle: number; lower: number } {
    const sma = calculateSMA(prices, period);
    const stdDev = calculateStandardDeviation(
        prices.slice(-period),
        sma
    );

    return {
        upper: sma + (stdDev * stdDevMultiplier),
        middle: sma,
        lower: sma - (stdDev * stdDevMultiplier)
    };
}

export function calculateRSI(prices: number[], period: number): number {
    let gains = 0;
    let losses = 0;

    for (let i = 1; i < prices.length; i++) {
        const difference = prices[i] - prices[i - 1];
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

export function calculateVolatility(
    prices: number[],
    period: number
): number {
    const returns = prices.slice(1).map((price, i) => 
        Math.log(price / prices[i])
    );
    return calculateStandardDeviation(returns, calculateMean(returns)) * 
           Math.sqrt(252); // Annualized
}

export function calculateSMA(prices: number[], period: number): number {
    return calculateMean(prices.slice(-period));
}

export function analyzeRevertSignal(
    stats: PriceStatistics,
    config: any
): RevertSignal {
    // Combine multiple indicators for reversion probability
    const zScoreSignal = Math.min(
        Math.abs(stats.zScore) / config.zScoreThreshold,
        1
    );
    
    const bbSignal = calculateBBSignal(stats.bollingerBands, 
        stats.mean);
    
    const rsiSignal = calculateRSISignal(stats.rsi, 
        config.rsiOverbought,
        config.rsiOversold);

    const strength = (zScoreSignal + bbSignal + rsiSignal) / 3;
    const direction = stats.zScore > 0 ? 'DOWN' : 'UP';
    
    // Calculate confidence based on volatility and signal agreement
    const confidence = calculateConfidence(stats, strength);
    
    // Estimate expected return based on historical mean reversion
    const expectedReturn = calculateExpectedReturn(stats, direction);

    return {
        strength,
        direction,
        confidence,
        expectedReturn
    };
}

function calculateBBSignal(
    bb: { upper: number; middle: number; lower: number },
    price: number
): number {
    const upperDist = Math.max(0, price - bb.upper) / (bb.upper - bb.middle);
    const lowerDist = Math.max(0, bb.lower - price) / (bb.middle - bb.lower);
    return Math.max(upperDist, lowerDist);
}

function calculateRSISignal(
    rsi: number,
    overbought: number,
    oversold: number
): number {
    if (rsi > overbought) {
        return (rsi - overbought) / (100 - overbought);
    }
    if (rsi < oversold) {
        return (oversold - rsi) / oversold;
    }
    return 0;
}

function calculateConfidence(
    stats: PriceStatistics,
    signalStrength: number
): number {
    const volatilityPenalty = Math.min(stats.volatility / 100, 0.5);
    return Math.max(0, signalStrength - volatilityPenalty);
}

function calculateExpectedReturn(
    stats: PriceStatistics,
    direction: 'UP' | 'DOWN'
): number {
    const expectedMove = stats.standardDeviation * 
        (direction === 'UP' ? 1 : -1);
    return expectedMove / stats.mean;
} 