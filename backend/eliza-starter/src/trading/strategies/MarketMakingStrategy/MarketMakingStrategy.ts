import { BaseStrategy } from '../BaseStrategy';
import { MarketData, StrategySignal, LimitOrderParams } from '../types';
import { MarketMakingConfig } from './types';
import { calculateVolatility, calculateOptimalSpread } from './utils';

export class MarketMakingStrategy extends BaseStrategy {
    private config: MarketMakingConfig;
    private positions: Map<string, number> = new Map();
    private openOrders: Map<string, LimitOrderParams[]> = new Map();
    private lastUpdate: number = 0;
    private dailyPnL: number = 0;

    constructor(config: Partial<MarketMakingConfig> = {}) {
        super('MarketMaking');
        
        // Default configuration
        this.config = {
            baseSpreadPercentage: 0.002,
            dynamicSpreadMultiplier: 1.5,
            minSpreadPercentage: 0.001,
            maxSpreadPercentage: 0.01,
            maxPositionSize: 1000,
            minOrderSize: 0.1,
            maxOrderSize: 10,
            maxDailyLoss: -1000,
            maxDrawdown: 0.1,
            minLiquidityUsd: 100000,
            orderRefreshTime: 30000,
            maxOpenOrders: 3,
            maxVolatility: 0.05,
            minVolumeUsd: 50000,
            ...config
        };
    }

    async analyzeMarket(marketData: MarketData): Promise<StrategySignal[]> {
        if (!this.validateMarketConditions(marketData)) {
            return [{ 
                action: 'HOLD',
                tokenAddress: '',
                amount: 0,
                notes: 'Market conditions not suitable for market making'
            }];
        }

        const signals: StrategySignal[] = [];
        const { orderBook } = marketData;
        
        if (!orderBook) {
            return signals;
        }

        const midPrice = (orderBook.bestBid + orderBook.bestAsk) / 2;
        const volatility = calculateVolatility(marketData);
        const spread = calculateOptimalSpread(
            this.config.baseSpreadPercentage,
            volatility,
            this.config.dynamicSpreadMultiplier,
            this.config.minSpreadPercentage,
            this.config.maxSpreadPercentage
        );

        // Calculate order sizes based on available liquidity and position
        const currentPosition = this.positions.get(marketData.tokenAddress) || 0;
        const buySize = this.calculateOrderSize(
            'BUY',
            currentPosition,
            marketData.liquidityUsd,
            midPrice
        );
        const sellSize = this.calculateOrderSize(
            'SELL',
            currentPosition,
            marketData.liquidityUsd,
            midPrice
        );

        // Generate buy and sell signals with limit prices
        if (buySize > 0) {
            signals.push({
                action: 'BUY',
                tokenAddress: marketData.tokenAddress,
                amount: buySize,
                price: midPrice * (1 - spread),
                orderType: 'LIMIT',
                confidence: 1 - (volatility / this.config.maxVolatility),
                notes: `Market making BUY order`,
                metadata: {
                    spread,
                    volatility,
                    midPrice
                }
            });
        }

        if (sellSize > 0) {
            signals.push({
                action: 'SELL',
                tokenAddress: marketData.tokenAddress,
                amount: sellSize,
                price: midPrice * (1 + spread),
                orderType: 'LIMIT',
                confidence: 1 - (volatility / this.config.maxVolatility),
                notes: `Market making SELL order`,
                metadata: {
                    spread,
                    volatility,
                    midPrice
                }
            });
        }

        return signals;
    }

    private validateMarketConditions(marketData: MarketData): boolean {
        if (!this.validateMarketData(marketData)) return false;

        return (
            marketData.liquidityUsd >= this.config.minLiquidityUsd &&
            marketData.volume24h >= this.config.minVolumeUsd &&
            (marketData.volatility24h || 0) <= this.config.maxVolatility &&
            this.dailyPnL >= -this.config.maxDailyLoss
        );
    }

    private calculateOrderSize(
        side: 'BUY' | 'SELL',
        currentPosition: number,
        liquidity: number,
        price: number
    ): number {
        const maxPositionValue = this.config.maxPositionSize * price;
        const currentPositionValue = currentPosition * price;
        
        let size: number;
        if (side === 'BUY') {
            size = Math.min(
                (maxPositionValue - currentPositionValue) / price,
                this.config.maxOrderSize
            );
        } else {
            size = Math.min(
                (maxPositionValue + currentPositionValue) / price,
                this.config.maxOrderSize
            );
        }

        return Math.max(size, this.config.minOrderSize);
    }

    updateState(updateParams: any): void {
        if (updateParams.position) {
            this.positions.set(
                updateParams.tokenAddress,
                updateParams.position
            );
        }

        if (updateParams.pnl) {
            this.dailyPnL = updateParams.pnl;
        }

        if (updateParams.orders) {
            this.openOrders.set(
                updateParams.tokenAddress,
                updateParams.orders
            );
        }
    }
} 