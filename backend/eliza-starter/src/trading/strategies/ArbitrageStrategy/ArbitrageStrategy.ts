import { BaseStrategy } from '../BaseStrategy';
import { MarketData, StrategySignal } from '../types';
import { 
    ArbitrageConfig, 
    ArbitrageOpportunity,
    ExchangeQuote 
} from './types';
import { 
    findSimpleArbitrage,
    findTriangularArbitrage 
} from './utils';

export class ArbitrageStrategy extends BaseStrategy {
    private config: ArbitrageConfig;
    private activeOpportunities: Map<string, ArbitrageOpportunity>;
    private executionHistory: {
        timestamp: number;
        opportunity: ArbitrageOpportunity;
        success: boolean;
        actualProfit?: number;
    }[];

    constructor(config: Partial<ArbitrageConfig> = {}) {
        super('Arbitrage');
        
        this.config = {
            minProfitUsd: 1,
            minProfitPercentage: 0.1, // 0.1%
            maxTradeSize: 1000,
            maxExposureUsd: 10000,
            minLiquidityUsd: 50000,
            maxExecutionTime: 5000,
            maxSlippage: 1,
            profitMonitoringInterval: 1000,
            maxConsecutiveLosses: 3,
            enabledExchanges: ['Jupiter', 'Raydium', 'Orca'],
            exchangePriority: ['Jupiter', 'Raydium', 'Orca'],
            ...config
        };

        this.activeOpportunities = new Map();
        this.executionHistory = [];
    }

    async analyzeMarket(marketData: MarketData): Promise<StrategySignal[]> {
        const signals: StrategySignal[] = [];
        
        // 1. Find simple arbitrage opportunities
        const quotes = new Map<string, ExchangeQuote>();
        for (const [exchange, data] of Object.entries(marketData.exchangeQuotes)) {
            if (this.config.enabledExchanges.includes(exchange)) {
                quotes.set(exchange, data as ExchangeQuote);
            }
        }

        const simpleArb = findSimpleArbitrage(
            quotes,
            this.config.minProfitUsd,
            this.config.minLiquidityUsd
        );

        // 2. Find triangular arbitrage opportunities
        const triangularArb = findTriangularArbitrage(
            marketData.pairs,
            marketData.tokenAddress,
            this.config.minProfitUsd
        );

        // 3. Select the most profitable opportunity
        const opportunities = [simpleArb, triangularArb]
            .filter(Boolean)
            .sort((a, b) => b!.expectedProfitUsd - a!.expectedProfitUsd);

        if (opportunities.length > 0) {
            const bestOpportunity = opportunities[0]!;
            
            // Generate signals for each step in the arbitrage path
            for (const step of bestOpportunity.path) {
                signals.push({
                    action: step.action,
                    tokenAddress: step.token,
                    amount: step.amount,
                    price: step.price,
                    orderType: 'MARKET',
                    confidence: bestOpportunity.confidence,
                    notes: `${bestOpportunity.type} arbitrage on ${step.exchange}`,
                    metadata: {
                        exchange: step.exchange,
                        expectedProfit: bestOpportunity.expectedProfitUsd,
                        arbitrageType: bestOpportunity.type
                    }
                });
            }

            // Track the opportunity
            this.activeOpportunities.set(
                `${marketData.tokenAddress}-${Date.now()}`,
                bestOpportunity
            );
        }

        return signals;
    }

    async executeTrades(
        signals: StrategySignal[],
        tradeManager: any
    ): Promise<void> {
        const startTime = Date.now();
        const results = [];

        for (const signal of signals) {
            try {
                if (Date.now() - startTime > this.config.maxExecutionTime) {
                    throw new Error('Execution timeout');
                }

                const result = await tradeManager.executeTrade({
                    inputToken: signal.action === 'BUY' ? 'USDC' : signal.tokenAddress,
                    outputToken: signal.action === 'BUY' ? signal.tokenAddress : 'USDC',
                    amount: signal.amount,
                    slippage: this.config.maxSlippage,
                    exchange: signal.metadata?.exchange
                });

                results.push(result);
            } catch (error) {
                console.error(`Arbitrage execution failed:`, error);
                // Implement cleanup/reversal logic if needed
                throw error;
            }
        }

        // Record execution results
        this.recordExecution(signals, results);
    }

    private recordExecution(
        signals: StrategySignal[],
        results: any[]
    ): void {
        const opportunity = signals[0]?.metadata?.arbitrageType === 'SIMPLE' ?
            'SIMPLE' : 'TRIANGULAR';
        
        this.executionHistory.push({
            timestamp: Date.now(),
            opportunity: this.activeOpportunities.get(
                `${signals[0].tokenAddress}-${Date.now()}`
            )!,
            success: results.every(r => r.success),
            actualProfit: this.calculateActualProfit(results)
        });
    }

    private calculateActualProfit(results: any[]): number {
        // Implementation depends on your execution results format
        return 0;
    }

    updateState(updateParams: any): void {
        super.updateState(updateParams);
        
        // Clean up old opportunities
        const now = Date.now();
        for (const [key, opportunity] of this.activeOpportunities) {
            if (now - parseInt(key.split('-')[1]) > this.config.maxExecutionTime) {
                this.activeOpportunities.delete(key);
            }
        }
    }
} 