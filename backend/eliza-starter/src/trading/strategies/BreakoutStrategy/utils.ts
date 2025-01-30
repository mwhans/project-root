import { PriceLevel, BreakoutConfig, BreakoutSignal } from './types';

export function findSupportResistanceLevels(
    prices: number[],
    config: BreakoutConfig
): PriceLevel[] {
    const levels: PriceLevel[] = [];
    const pricePoints = findPivotPoints(prices);

    // Group similar price levels
    const clusters = clusterPriceLevels(pricePoints, 0.005); // 0.5% tolerance

    // Convert clusters to support/resistance levels
    clusters.forEach(cluster => {
        const avgPrice = calculateClusterAverage(cluster);
        const touches = countTouches(prices, avgPrice, 0.005);
        
        if (touches >= config.minTouchPoints) {
            const strength = calculateLevelStrength(
                cluster,
                touches,
                prices.length
            );

            if (strength >= config.levelStrengthThreshold) {
                levels.push({
                    price: avgPrice,
                    strength,
                    touches,
                    lastTouch: findLastTouch(prices, avgPrice),
                    type: determineLevelType(avgPrice, prices[prices.length - 1])
                });
            }
        }
    });

    return levels;
}

function findPivotPoints(prices: number[]): number[] {
    const pivots: number[] = [];
    const window = 5; // Look 5 periods before/after

    for (let i = window; i < prices.length - window; i++) {
        if (isHighPivot(prices, i, window)) {
            pivots.push(prices[i]);
        }
        if (isLowPivot(prices, i, window)) {
            pivots.push(prices[i]);
        }
    }

    return pivots;
}

function clusterPriceLevels(
    prices: number[],
    tolerance: number
): number[][] {
    const clusters: number[][] = [];
    
    prices.forEach(price => {
        let added = false;
        for (const cluster of clusters) {
            if (Math.abs(cluster[0] - price) / cluster[0] <= tolerance) {
                cluster.push(price);
                added = true;
                break;
            }
        }
        if (!added) {
            clusters.push([price]);
        }
    });

    return clusters;
}

export function detectBreakout(
    currentPrice: number,
    levels: PriceLevel[],
    volume: number,
    averageVolume: number,
    rsi: number,
    config: BreakoutConfig
): BreakoutSignal | null {
    // Find the nearest level that's been broken
    const brokenLevel = findBrokenLevel(currentPrice, levels);
    if (!brokenLevel) return null;

    // Calculate breakout strength and confirmations
    const volumeConfirmed = volume > averageVolume * config.minVolumeIncrease;
    const rsiConfirmed = confirmRSI(
        rsi,
        brokenLevel.type,
        config.rsiThresholds
    );
    
    const strength = calculateBreakoutStrength(
        currentPrice,
        brokenLevel,
        volumeConfirmed,
        rsiConfirmed
    );

    const expectedMove = calculateExpectedMove(
        currentPrice,
        brokenLevel,
        levels
    );

    return {
        direction: brokenLevel.type === 'RESISTANCE' ? 'UP' : 'DOWN',
        strength,
        level: brokenLevel,
        confirmation: {
            volume: volumeConfirmed,
            rsi: rsiConfirmed,
            price: true
        },
        expectedMove
    };
}

function calculateBreakoutStrength(
    currentPrice: number,
    level: PriceLevel,
    volumeConfirmed: boolean,
    rsiConfirmed: boolean
): number {
    let strength = level.strength;
    
    // Add volume confirmation bonus
    if (volumeConfirmed) strength *= 1.2;
    
    // Add RSI confirmation bonus
    if (rsiConfirmed) strength *= 1.1;
    
    // Add breakout distance factor
    const breakoutPercent = Math.abs(currentPrice - level.price) / level.price;
    strength *= (1 + breakoutPercent);

    return Math.min(strength, 1);
}

function calculateExpectedMove(
    currentPrice: number,
    brokenLevel: PriceLevel,
    levels: PriceLevel[]
): number {
    // Find next significant level in breakout direction
    const nextLevel = findNextLevel(currentPrice, brokenLevel.type, levels);
    if (nextLevel) {
        return Math.abs(nextLevel.price - brokenLevel.price);
    }
    
    // If no next level, use average historical move
    return Math.abs(currentPrice - brokenLevel.price) * 1.5;
} 