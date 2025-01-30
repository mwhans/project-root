import { BaseStrategy } from '../BaseStrategy';
import { MarketData, StrategySignal } from '../types';
import { 
    CombinerConfig,
    StrategyPerformance,
    CombinedSignal 
} from './types';
import {
    combineSignals,
    updateStrategyPerformance
} from './utils';

export class StrategyCombiner extends BaseStrategy {
    private config: CombinerConfig;
    private strategies: BaseStrategy[];
    private performance: Map<string, StrategyPerformance>;
    private lastExecutions: Map<string, number>;
    private activeSignals: Map<string, CombinedSignal>;

    constructor(
        strategies: BaseStrategy[],
        config: Partial<CombinerConfig> = {}
    ) {
        super('StrategyCombiner');
        
        this.strategies = strategies;
        this.config = {
            combinationMethod: 'WEIGHTED_VOTE',
            conflictResolution: 'WEIGHTED_AVERAGE',
            strategyWeights: strategies.map(s => ({
                strategyName: s.strategyName,
                weight: 1 / strategies.length
            })),
            minStrategyAgreement: 0.51,
            minConfidenceThreshold: 0.6,
            minSignalStrength: 0.5,
            maxPositionSize: 1000,
            maxSimultaneousSignals: 3,
            riskPerTrade: 0.02,
            performanceWindow: 7 * 24 * 60 * 60 * 1000, // 1 week
            adaptiveWeights: true,
            sequentialExecution: false,
            executionTimeout: 30000,
            retryAttempts: 3,
            ...config
        };

        this.performance = new Map(
            strategies.map(s => [
                s.strategyName,
                {
                    strategyName: s.strategyName,
                    signals: 0,
                    successRate: 0,
                    profitFactor: 1,
                    sharpeRatio: 0,
                    weight: this.config.strategyWeights.find(
                        w => w.strategyName === s.strategyName
                    )?.weight || 1/strategies.length,
                    lastUpdated: Date.now()
                }
            ])
        );

        this.lastExecutions = new Map();
        this.activeSignals = new Map();
    }

    async analyzeMarket(marketData: MarketData): Promise<StrategySignal[]> {
        // Collect signals from all strategies
        const allSignals: StrategySignal[] = [];
        
        for (const strategy of this.strategies) {
            try {
                const signals = await strategy.analyzeMarket(marketData);
                signals.forEach(s => s.strategyName = strategy.strategyName);
                allSignals.push(...signals);
            } catch (error) {
                console.error(
                    `Error in strategy ${strategy.strategyName}:`,
                    error
                );
            }
        }

        // Combine signals using configured method
        const combinedSignals = combineSignals(
            allSignals,
            this.config,
            this.performance
        );

        // Apply risk management
        const finalSignals = this.applyRiskManagement(
            combinedSignals,
            marketData
        );

        // Update active signals
        finalSignals.forEach(signal => {
            this.activeSignals.set(
                signal.combined.tokenAddress,
                signal
            );
        });

        return finalSignals.map(s => s.combined);
    }

    async executeTrades(
        signals: StrategySignal[],
        tradeManager: any
    ): Promise<void> {
        if (this.config.sequentialExecution) {
            await this.executeSequentially(signals, tradeManager);
        } else {
            await this.executeParallel(signals, tradeManager);
        }
    }

    private async executeSequentially(
        signals: StrategySignal[],
        tradeManager: any
    ): Promise<void> {
        for (const signal of signals) {
            const combinedSignal = this.activeSignals.get(
                signal.tokenAddress
            );
            if (!combinedSignal) continue;

            for (const strategy of this.strategies) {
                if (combinedSignal.strategies.includes(
                    strategy.strategyName
                )) {
                    try {
                        await strategy.executeTrades(
                            [signal],
                            tradeManager
                        );
                        this.lastExecutions.set(
                            strategy.strategyName,
                            Date.now()
                        );
                    } catch (error) {
                        console.error(
                            `Error executing ${strategy.strategyName}:`,
                            error
                        );
                    }
                }
            }
        }
    }

    private async executeParallel(
        signals: StrategySignal[],
        tradeManager: any
    ): Promise<void> {
        const executions = signals.map(async signal => {
            const combinedSignal = this.activeSignals.get(
                signal.tokenAddress
            );
            if (!combinedSignal) return;

            const strategyExecutions = this.strategies
                .filter(s => combinedSignal.strategies.includes(
                    s.strategyName
                ))
                .map(strategy => 
                    this.executeWithRetry(
                        strategy,
                        signal,
                        tradeManager
                    )
                );

            await Promise.all(strategyExecutions);
        });

        await Promise.all(executions);
    }

    private async executeWithRetry(
        strategy: BaseStrategy,
        signal: StrategySignal,
        tradeManager: any
    ): Promise<void> {
        for (let i = 0; i < this.config.retryAttempts; i++) {
            try {
                await strategy.executeTrades([signal], tradeManager);
                this.lastExecutions.set(strategy.strategyName, Date.now());
                break;
            } catch (error) {
                if (i === this.config.retryAttempts - 1) {
                    throw error;
                }
                await new Promise(r => setTimeout(
                    r,
                    Math.pow(2, i) * 1000
                ));
            }
        }
    }

    private applyRiskManagement(
        signals: CombinedSignal[],
        marketData: MarketData
    ): CombinedSignal[] {
        return signals
            .filter(signal => 
                signal.confidence >= this.config.minConfidenceThreshold &&
                signal.agreement >= this.config.minStrategyAgreement
            )
            .sort((a, b) => b.confidence - a.confidence)
            .slice(0, this.config.maxSimultaneousSignals)
            .map(signal => ({
                ...signal,
                combined: {
                    ...signal.combined,
                    amount: this.calculateRiskAdjustedAmount(
                        signal,
                        marketData
                    )
                }
            }));
    }

    private calculateRiskAdjustedAmount(
        signal: CombinedSignal,
        marketData: MarketData
    ): number {
        const baseAmount = signal.combined.amount || 0;
        const riskAdjustment = signal.confidence * 
            this.config.riskPerTrade;
        
        return Math.min(
            baseAmount * riskAdjustment,
            this.config.maxPositionSize
        );
    }

    updateState(updateParams: any): void {
        super.updateState(updateParams);
        
        // Update child strategies
        this.strategies.forEach(strategy => {
            strategy.updateState(updateParams);
        });

        // Update performance metrics
        if (updateParams.tradeResults) {
            updateStrategyPerformance(
                this.performance,
                updateParams.tradeResults,
                this.config
            );
        }

        // Clean up old signals
        this.cleanupOldSignals();
    }

    private cleanupOldSignals(): void {
        const now = Date.now();
        for (const [token, signal] of this.activeSignals) {
            if (now - signal.combined.timestamp! > 
                this.config.executionTimeout) {
                this.activeSignals.delete(token);
            }
        }
    }
} 