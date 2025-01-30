import { ITradingStrategy, MarketData, StrategySignal } from './types';
import { TradeManager } from '../TradeManager';

export abstract class BaseStrategy implements ITradingStrategy {
    protected constructor(
        public readonly strategyName: string,
        protected readonly config: Record<string, any> = {}
    ) {}

    abstract analyzeMarket(marketData: MarketData): Promise<StrategySignal[]>;

    async executeTrades(signals: StrategySignal[], tradeManager: TradeManager): Promise<void> {
        for (const signal of signals) {
            if (signal.action === 'HOLD') continue;

            try {
                const params = {
                    inputToken: signal.action === 'BUY' ? 'USDC' : signal.tokenAddress,
                    outputToken: signal.action === 'BUY' ? signal.tokenAddress : 'USDC',
                    amount: signal.amount,
                    slippage: 100, // 1% default slippage
                };

                await tradeManager.executeTrade(params);
            } catch (error) {
                console.error(`Failed to execute ${signal.action} for ${signal.tokenAddress}:`, error);
                throw error;
            }
        }
    }

    updateState(updateParams: any): void {
        // Default implementation - override as needed
        Object.assign(this.config, updateParams);
    }

    protected validateMarketData(data: MarketData): boolean {
        return !!(
            data &&
            typeof data.price === 'number' &&
            typeof data.volume24h === 'number' &&
            typeof data.liquidityUsd === 'number'
        );
    }
} 