import { GridLevel, GridConfig, GridState } from './types';

export function calculateGridLevels(
    referencePrice: number,
    config: GridConfig,
    volatility?: number
): GridLevel[] {
    const levels: GridLevel[] = [];
    const spacing = config.dynamicSpacing && volatility ?
        adjustGridSpacing(config.gridSpacing, volatility) :
        config.gridSpacing;

    // Calculate buy levels below reference
    for (let i = 1; i <= config.gridLevels; i++) {
        const price = referencePrice * (1 - i * spacing);
        if (config.lowerBoundary && price < config.lowerBoundary) break;

        levels.push({
            price,
            side: 'BUY',
            size: calculateLevelSize(i, config),
            status: 'PENDING',
            timestamp: Date.now()
        });
    }

    // Calculate sell levels above reference
    for (let i = 1; i <= config.gridLevels; i++) {
        const price = referencePrice * (1 + i * spacing);
        if (config.upperBoundary && price > config.upperBoundary) break;

        levels.push({
            price,
            side: 'SELL',
            size: calculateLevelSize(i, config),
            status: 'PENDING',
            timestamp: Date.now()
        });
    }

    return levels;
}

function adjustGridSpacing(
    baseSpacing: number,
    volatility: number
): number {
    // Adjust grid spacing based on market volatility
    const volatilityFactor = Math.min(volatility * 10, 2);
    return baseSpacing * volatilityFactor;
}

function calculateLevelSize(
    levelIndex: number,
    config: GridConfig
): number {
    // Increase size with distance from reference price
    return config.baseOrderSize * 
           Math.pow(config.sizeMultiplier, levelIndex - 1);
}

export function shouldRebalanceGrid(
    currentPrice: number,
    gridState: GridState,
    config: GridConfig
): boolean {
    const priceDeviation = Math.abs(
        currentPrice - gridState.referencePrice
    ) / gridState.referencePrice;

    const timeElapsed = Date.now() - gridState.lastRebalance;

    return (
        priceDeviation > config.rebalanceThreshold ||
        timeElapsed > config.rebalanceInterval
    );
}

export function calculateGridProfitability(
    gridState: GridState,
    currentPrice: number
): {
    unrealizedProfit: number;
    realizedProfit: number;
    roi: number;
} {
    let unrealizedProfit = 0;
    const realizedProfit = gridState.currentProfit;

    // Calculate unrealized profit from active positions
    gridState.levels
        .filter(level => level.status === 'FILLED')
        .forEach(level => {
            if (level.side === 'BUY') {
                unrealizedProfit += (currentPrice - level.price) * level.size;
            } else {
                unrealizedProfit += (level.price - currentPrice) * level.size;
            }
        });

    const roi = (realizedProfit + unrealizedProfit) / 
                gridState.totalInvestment * 100;

    return { unrealizedProfit, realizedProfit, roi };
} 