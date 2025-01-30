import { BaseStrategy } from '../BaseStrategy';
import { MarketData, StrategySignal } from '../types';
import { 
    MeanReversionConfig,
    PriceStatistics,
    RevertSignal 
} from './types';
import {
    calculatePriceStatistics,
    analyzeRevertSignal
} from './utils';

export class MeanReversionStrategy extends BaseStrategy {
    private config: MeanReversionConfig;
    private priceStats: PriceStatistics;
    private positions: Map<string, {
        entryPrice: number;
        size: number;
        stopLoss: number;
        takeProfit: number;
        entryTime: number;
    }>;

    constructor(config: Partial<MeanReversionConfig> = {}) {
        super('MeanReversion');
        
        this.config = {
            lookbackPeriod: 20,
            bollingerBandPeriod: 20,
            bollingerBandStdDev: 2,
            rsiPeriod: 14,
            zScoreThreshold: 2,
            rsiOverbought: 70,
            rsiOversold: 30,
            maxPositionSize: 1000,
            positionSizeScaling: 0.5,
            stopLossDeviation: 3,
            takeProfitDeviation: 1,
            maxDrawdown: 10,
            maxPositions: 3,
            minVolume: 100000,
            maxSpread: 0.5,
            entryTimeout: 60000,
            exitTimeout: 30000,
            minProfitTarget: 0.5,
            ...config
        };

        this.priceStats = {} as PriceStatistics;
        this.positions = new Map();
    }

    async analyzeMarket(marketData: MarketData): Promise<StrategySignal[]> {
        if (!this.validateMarketConditions(marketData)) {
            return [{
                action: 'HOLD',
                tokenAddress: marketData.tokenAddress,
                amount: 0,
                notes: 'Market conditions not suitable for mean reversion'
            }];
        }

        // Update price statistics
        this.priceStats = calculatePriceStatistics(
            marketData.priceHistory,
            this.config.lookbackPeriod
        );

        // Analyze reversion signals
        const revertSignal = analyzeRevertSignal(this.priceStats, this.config);

        // Generate trading signals
        return this.generateSignals(marketData, revertSignal);
    }

    private validateMarketConditions(marketData: MarketData): boolean {
        return (
            marketData.volume24h >= this.config.minVolume &&
            marketData.liquidityUsd >= this.config.minVolume &&
            (marketData.orderBook?.spread || 0) <= this.config.maxSpread
        );
    }

    private generateSignals(
        marketData: MarketData,
        revertSignal: RevertSignal
    ): StrategySignal[] {
        const signals: StrategySignal[] = [];
        const currentPosition = this.positions.get(marketData.tokenAddress);

        if (currentPosition) {
            if (this.shouldExitPosition(marketData, currentPosition)) {
                signals.push(this.generateExitSignal(marketData, currentPosition));
            }
        } else if (
            revertSignal.strength >= 0.7 &&
            revertSignal.confidence >= 0.6 &&
            Math.abs(revertSignal.expectedReturn) >= this.config.minProfitTarget
        ) {
            signals.push(
                this.generateEntrySignal(marketData, revertSignal)
            );
        }

        return signals;
    }

    private shouldExitPosition(
        marketData: MarketData,
        position: any
    ): boolean {
        const currentPrice = marketData.price;
        const timeSinceEntry = Date.now() - position.entryTime;
        
        return (
            currentPrice <= position.stopLoss ||
            currentPrice >= position.takeProfit ||
            timeSinceEntry >= this.config.exitTimeout ||
            Math.abs(this.priceStats.zScore) <= 0.5
        );
    }

    private generateEntrySignal(
        marketData: MarketData,
        revertSignal: RevertSignal
    ): StrategySignal {
        const positionSize = this.calculatePositionSize(
            revertSignal.strength,
            marketData.price
        );

        const stopLoss = marketData.price * (1 + (
            revertSignal.direction === 'UP' ? -1 : 1
        ) * (this.priceStats.standardDeviation * 
             this.config.stopLossDeviation / marketData.price));

        const takeProfit = this.priceStats.mean;

        this.positions.set(marketData.tokenAddress, {
            entryPrice: marketData.price,
            size: positionSize,
            stopLoss,
            takeProfit,
            entryTime: Date.now()
        });

        return {
            action: revertSignal.direction === 'UP' ? 'BUY' : 'SELL',
            tokenAddress: marketData.tokenAddress,
            amount: positionSize,
            price: marketData.price,
            orderType: 'LIMIT',
            confidence: revertSignal.confidence,
            notes: `Mean reversion ${revertSignal.direction} signal`,
            metadata: {
                zScore: this.priceStats.zScore,
                expectedReturn: revertSignal.expectedReturn,
                stopLoss,
                takeProfit,
                statistics: this.priceStats
            }
        };
    }

    private generateExitSignal(
        marketData: MarketData,
        position: any
    ): StrategySignal {
        return {
            action: 'SELL',
            tokenAddress: marketData.tokenAddress,
            amount: position.size,
            price: marketData.price,
            orderType: 'MARKET',
            confidence: 1,
            notes: 'Mean reversion exit',
            metadata: {
                exitReason: this.getExitReason(marketData, position),
                profitLoss: (marketData.price - position.entryPrice) * 
                           position.size,
                holdingTime: Date.now() - position.entryTime
            }
        };
    }

    private calculatePositionSize(
        signalStrength: number,
        currentPrice: number
    ): number {
        const baseSize = this.config.maxPositionSize / currentPrice;
        return baseSize * signalStrength * this.config.positionSizeScaling;
    }

    private getExitReason(
        marketData: MarketData,
        position: any
    ): string {
        if (marketData.price <= position.stopLoss) return 'Stop loss';
        if (marketData.price >= position.takeProfit) return 'Take profit';
        if (Date.now() - position.entryTime >= this.config.exitTimeout) {
            return 'Timeout';
        }
        return 'Mean reversion complete';
    }

    updateState(updateParams: any): void {
        super.updateState(updateParams);
        
        // Update position tracking
        if (updateParams.closedPositions) {
            for (const position of updateParams.closedPositions) {
                this.positions.delete(position.tokenAddress);
            }
        }
    }
} 