import { 
    CombinerConfig,
    StrategyWeight,
    CombinedSignal,
    StrategyPerformance
} from './types';
import { StrategySignal } from '../types';

export function combineSignals(
    signals: StrategySignal[],
    config: CombinerConfig,
    performance: Map<string, StrategyPerformance>
): CombinedSignal[] {
    const groupedSignals = groupSignalsByToken(signals);
    const combinedSignals: CombinedSignal[] = [];

    for (const [token, tokenSignals] of groupedSignals) {
        const combined = applyCombinationMethod(
            tokenSignals,
            config,
            performance
        );
        if (combined) combinedSignals.push(combined);
    }

    return filterAndPrioritizeSignals(combinedSignals, config);
}

function groupSignalsByToken(
    signals: StrategySignal[]
): Map<string, StrategySignal[]> {
    const grouped = new Map<string, StrategySignal[]>();
    
    signals.forEach(signal => {
        if (!grouped.has(signal.tokenAddress)) {
            grouped.set(signal.tokenAddress, []);
        }
        grouped.get(signal.tokenAddress)!.push(signal);
    });

    return grouped;
}

function applyCombinationMethod(
    signals: StrategySignal[],
    config: CombinerConfig,
    performance: Map<string, StrategyPerformance>
): CombinedSignal | null {
    switch (config.combinationMethod) {
        case 'MAJORITY_VOTE':
            return applyMajorityVote(signals, config);
        case 'WEIGHTED_VOTE':
            return applyWeightedVote(signals, config, performance);
        case 'CONSENSUS':
            return applyConsensus(signals, config);
        case 'PRIORITY':
            return applyPriority(signals, config);
        case 'SEQUENTIAL':
            return applySequential(signals, config);
        default:
            throw new Error(
                `Unknown combination method: ${config.combinationMethod}`
            );
    }
}

function applyWeightedVote(
    signals: StrategySignal[],
    config: CombinerConfig,
    performance: Map<string, StrategyPerformance>
): CombinedSignal | null {
    let weightedBuy = 0;
    let weightedSell = 0;
    let totalWeight = 0;

    signals.forEach(signal => {
        const weight = getStrategyWeight(
            signal.strategyName!,
            config,
            performance
        );
        const signalWeight = weight * (signal.confidence || 1);
        
        if (signal.action === 'BUY') weightedBuy += signalWeight;
        if (signal.action === 'SELL') weightedSell += signalWeight;
        totalWeight += weight;
    });

    const agreement = Math.max(
        weightedBuy, weightedSell
    ) / totalWeight;

    if (agreement < config.minStrategyAgreement) return null;

    const action = weightedBuy > weightedSell ? 'BUY' : 'SELL';
    const confidence = Math.abs(weightedBuy - weightedSell) / totalWeight;

    return {
        original: signals,
        combined: {
            action,
            tokenAddress: signals[0].tokenAddress,
            amount: calculateCombinedAmount(signals, config),
            confidence,
            notes: `Combined ${signals.length} signals with ${
                config.combinationMethod
            }`,
            metadata: {
                agreement,
                strategies: signals.map(s => s.strategyName!)
            }
        },
        confidence,
        agreement,
        strategies: signals.map(s => s.strategyName!)
    };
}

function calculateCombinedAmount(
    signals: StrategySignal[],
    config: CombinerConfig
): number {
    const baseAmount = signals.reduce(
        (sum, signal) => sum + (signal.amount || 0),
        0
    ) / signals.length;

    return Math.min(
        baseAmount,
        config.maxPositionSize
    );
}

export function updateStrategyPerformance(
    performance: Map<string, StrategyPerformance>,
    results: any[],
    config: CombinerConfig
): void {
    results.forEach(result => {
        const perf = performance.get(result.strategyName);
        if (perf) {
            // Update performance metrics
            perf.signals++;
            perf.successRate = updateSuccessRate(
                perf.successRate,
                result.success,
                perf.signals
            );
            perf.profitFactor = updateProfitFactor(
                perf.profitFactor,
                result.profit
            );
            perf.lastUpdated = Date.now();

            // Update weight if adaptive weights enabled
            if (config.adaptiveWeights) {
                perf.weight = calculateAdaptiveWeight(perf);
            }
        }
    });
}

function calculateAdaptiveWeight(
    performance: StrategyPerformance
): number {
    return (
        performance.successRate * 0.4 +
        Math.min(performance.profitFactor, 2) * 0.4 +
        Math.min(performance.sharpeRatio, 2) * 0.2
    );
} 