import { BaseStrategy } from '../BaseStrategy';
import { MarketData, StrategySignal } from '../types';
import { 
    BreakoutConfig, 
    PriceLevel, 
    BreakoutSignal 
} from './types';
import {
    findSupportResistanceLevels,
    detectBreakout
} from './utils';

export class BreakoutStrategy extends BaseStrategy {
    private config: BreakoutConfig;
    private levels: PriceLevel[];
    private activeBreakouts: Map<string, {
        signal: BreakoutSignal;
        entryPrice: number;
        stopLoss: number;
        takeProfit: number;
        size: number;
        timestamp: number;
    }>;

    constructor(config: Partial<BreakoutConfig> = {}) {
        super('Breakout');
        
        this.config = {
            lookbackPeriods: 100,
            minTouchPoints: 3,
            levelStrengthThreshold: 0.6,
            volumeConfirmation: true,
            minVolumeIncrease: 1.5,
            rsiConfirmation: true,
            rsiThresholds: {
                overbought: 70,
                oversold: 30
            },
            positionSize: 100,
            scalingFactor: 1.5,
            stopLossPercent: 2,
            takeProfitPercent: 6,
            maxPositions: 3,
            trailingStop: true,
            trailingStopDistance: 3,
            minBreakoutPercent: 0.5,
            confirmationPeriods: 3,
            falseBreakoutGuard: true,
            ...config
        };

        this.levels = [];
        this.activeBreakouts = new Map();
    }

    async analyzeMarket(marketData: MarketData): Promise<StrategySignal[]> {
        const signals: StrategySignal[] = [];
        
        // Update support/resistance levels
        this.updateLevels(marketData);
        
        // Check for breakouts
        const breakout = detectBreakout(
            marketData.price,
            this.levels,
            marketData.volume24h,
            marketData.averageVolume24h || 0,
            marketData.indicators?.rsi || 50,
            this.config
        );

        if (breakout && this.validateBreakout(breakout, marketData)) {
            signals.push(this.generateBreakoutSignal(breakout, marketData));
        }

        // Manage existing positions
        this.activeBreakouts.forEach((position, tokenAddress) => {
            const exitSignal = this.checkExitConditions(
                position,
                marketData
            );
            if (exitSignal) signals.push(exitSignal);
        });

        return signals;
    }

    private updateLevels(marketData: MarketData): void {
        if (marketData.priceHistory.length >= this.config.lookbackPeriods) {
            this.levels = findSupportResistanceLevels(
                marketData.priceHistory,
                this.config
            );
        }
    }

    private validateBreakout(
        breakout: BreakoutSignal,
        marketData: MarketData
    ): boolean {
        if (!this.config.falseBreakoutGuard) return true;

        // Check if we have enough confirmations
        const confirmations = [
            breakout.confirmation.volume,
            breakout.confirmation.rsi,
            breakout.confirmation.price
        ].filter(Boolean).length;

        // Check position limits
        if (this.activeBreakouts.size >= this.config.maxPositions) {
            return false;
        }

        // Validate breakout percentage
        const breakoutPercent = Math.abs(
            marketData.price - breakout.level.price
        ) / breakout.level.price;

        return (
            confirmations >= 2 &&
            breakoutPercent >= this.config.minBreakoutPercent &&
            breakout.strength >= this.config.levelStrengthThreshold
        );
    }

    private generateBreakoutSignal(
        breakout: BreakoutSignal,
        marketData: MarketData
    ): StrategySignal {
        const positionSize = this.calculatePositionSize(
            breakout.strength,
            marketData.price
        );

        const stopLoss = this.calculateStopLoss(
            marketData.price,
            breakout
        );

        const takeProfit = this.calculateTakeProfit(
            marketData.price,
            breakout
        );

        // Record the breakout
        this.activeBreakouts.set(marketData.tokenAddress, {
            signal: breakout,
            entryPrice: marketData.price,
            stopLoss,
            takeProfit,
            size: positionSize,
            timestamp: Date.now()
        });

        return {
            action: breakout.direction === 'UP' ? 'BUY' : 'SELL',
            tokenAddress: marketData.tokenAddress,
            amount: positionSize,
            price: marketData.price,
            orderType: 'MARKET',
            confidence: breakout.strength,
            notes: `Breakout ${breakout.direction} from ${breakout.level.type}`,
            metadata: {
                breakoutStrength: breakout.strength,
                stopLoss,
                takeProfit,
                expectedMove: breakout.expectedMove,
                confirmations: breakout.confirmation
            }
        };
    }

    private checkExitConditions(
        position: any,
        marketData: MarketData
    ): StrategySignal | null {
        const currentPrice = marketData.price;
        
        // Check stop loss
        if (currentPrice <= position.stopLoss) {
            return this.generateExitSignal(
                'STOP_LOSS',
                position,
                marketData
            );
        }

        // Check take profit
        if (currentPrice >= position.takeProfit) {
            return this.generateExitSignal(
                'TAKE_PROFIT',
                position,
                marketData
            );
        }

        // Update trailing stop if enabled
        if (
            this.config.trailingStop &&
            currentPrice > position.entryPrice
        ) {
            const newStopLoss = currentPrice * 
                (1 - this.config.trailingStopDistance / 100);
            if (newStopLoss > position.stopLoss) {
                position.stopLoss = newStopLoss;
            }
        }

        return null;
    }

    private generateExitSignal(
        reason: string,
        position: any,
        marketData: MarketData
    ): StrategySignal {
        this.activeBreakouts.delete(marketData.tokenAddress);

        return {
            action: 'SELL',
            tokenAddress: marketData.tokenAddress,
            amount: position.size,
            price: marketData.price,
            orderType: 'MARKET',
            confidence: 1,
            notes: `Breakout exit: ${reason}`,
            metadata: {
                exitReason: reason,
                profitLoss: (
                    marketData.price - position.entryPrice
                ) * position.size,
                holdingTime: Date.now() - position.timestamp
            }
        };
    }

    private calculatePositionSize(
        strength: number,
        currentPrice: number
    ): number {
        const baseSize = this.config.positionSize / currentPrice;
        return baseSize * (1 + (strength * this.config.scalingFactor));
    }

    private calculateStopLoss(
        entryPrice: number,
        breakout: BreakoutSignal
    ): number {
        const direction = breakout.direction === 'UP' ? 1 : -1;
        return entryPrice * (1 - direction * this.config.stopLossPercent / 100);
    }

    private calculateTakeProfit(
        entryPrice: number,
        breakout: BreakoutSignal
    ): number {
        const direction = breakout.direction === 'UP' ? 1 : -1;
        return entryPrice * (1 + direction * this.config.takeProfitPercent / 100);
    }

    updateState(updateParams: any): void {
        super.updateState(updateParams);
        
        // Update position management
        if (updateParams.closedPositions) {
            updateParams.closedPositions.forEach((position: any) => {
                this.activeBreakouts.delete(position.tokenAddress);
            });
        }
    }
} 