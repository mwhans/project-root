import { BaseStrategy } from '../BaseStrategy';
import { MarketData, StrategySignal } from '../types';
import { 
    TechnicalIndicators, 
    TrendFollowingConfig,
    TrendState 
} from './types';
import {
    calculateSMA,
    calculateEMA,
    calculateMACD,
    calculateRSI,
    calculateATR,
    analyzeTrendState
} from './utils';

export class TrendFollowingStrategy extends BaseStrategy {
    private config: TrendFollowingConfig;
    private indicators: TechnicalIndicators;
    private trendState: TrendState;
    private positions: Map<string, {
        entryPrice: number;
        size: number;
        stopLoss: number;
        takeProfit: number;
    }>;

    constructor(config: Partial<TrendFollowingConfig> = {}) {
        super('TrendFollowing');
        
        this.config = {
            fastMaPeriod: 10,
            slowMaPeriod: 21,
            rsiPeriod: 14,
            macdConfig: {
                fastPeriod: 12,
                slowPeriod: 26,
                signalPeriod: 9
            },
            rsiOverbought: 70,
            rsiOversold: 30,
            minTrendStrength: 0.3,
            minConfirmation: 0.6,
            maxPositionSize: 1000,
            positionSizeAdjustment: 0.5,
            stopLossPercentage: 2,
            takeProfitPercentage: 6,
            maxDrawdown: 10,
            trailingStopDistance: 3,
            ...config
        };

        this.indicators = {} as TechnicalIndicators;
        this.trendState = {} as TrendState;
        this.positions = new Map();
    }

    async analyzeMarket(marketData: MarketData): Promise<StrategySignal[]> {
        const signals: StrategySignal[] = [];
        
        // Update technical indicators
        this.updateIndicators(marketData);
        
        // Analyze trend state
        this.trendState = analyzeTrendState(this.indicators, this.config);
        
        // Generate trading signals based on trend analysis
        if (this.shouldEnterPosition(marketData)) {
            signals.push(this.generateEntrySignal(marketData));
        }
        
        if (this.shouldExitPosition(marketData)) {
            signals.push(this.generateExitSignal(marketData));
        }

        return signals;
    }

    private updateIndicators(marketData: MarketData): void {
        const prices = marketData.priceHistory;
        
        this.indicators = {
            sma: {
                fast: calculateSMA(prices, this.config.fastMaPeriod),
                slow: calculateSMA(prices, this.config.slowMaPeriod)
            },
            ema: {
                fast: calculateEMA(prices, this.config.fastMaPeriod),
                slow: calculateEMA(prices, this.config.slowMaPeriod)
            },
            macd: calculateMACD(
                prices,
                this.config.macdConfig.fastPeriod,
                this.config.macdConfig.slowPeriod,
                this.config.macdConfig.signalPeriod
            ),
            rsi: calculateRSI(prices, this.config.rsiPeriod),
            atr: calculateATR(
                marketData.highPrices,
                marketData.lowPrices,
                marketData.closePrices,
                this.config.slowMaPeriod
            ),
            volume: {
                current: marketData.volume24h,
                average: calculateSMA(marketData.volumeHistory, this.config.slowMaPeriod)
            }
        };
    }

    private shouldEnterPosition(marketData: MarketData): boolean {
        const currentPosition = this.positions.get(marketData.tokenAddress);
        if (currentPosition) return false;

        return (
            this.trendState.strength >= this.config.minTrendStrength &&
            this.trendState.confirmation >= this.config.minConfirmation &&
            ((this.trendState.direction === 'BULLISH' && this.indicators.rsi < this.config.rsiOverbought) ||
             (this.trendState.direction === 'BEARISH' && this.indicators.rsi > this.config.rsiOversold))
        );
    }

    private shouldExitPosition(marketData: MarketData): boolean {
        const position = this.positions.get(marketData.tokenAddress);
        if (!position) return false;

        const currentPrice = marketData.price;
        const stopLossHit = currentPrice <= position.stopLoss;
        const takeProfitHit = currentPrice >= position.takeProfit;
        const trendReversal = this.trendState.direction !== 'BULLISH' && 
                             this.trendState.confirmation >= this.config.minConfirmation;

        return stopLossHit || takeProfitHit || trendReversal;
    }

    private generateEntrySignal(marketData: MarketData): StrategySignal {
        const positionSize = this.calculatePositionSize(marketData);
        const entryPrice = marketData.price;
        
        // Calculate stop loss and take profit levels
        const stopLoss = entryPrice * (1 - this.config.stopLossPercentage / 100);
        const takeProfit = entryPrice * (1 + this.config.takeProfitPercentage / 100);

        // Store position details
        this.positions.set(marketData.tokenAddress, {
            entryPrice,
            size: positionSize,
            stopLoss,
            takeProfit
        });

        return {
            action: 'BUY',
            tokenAddress: marketData.tokenAddress,
            amount: positionSize,
            price: entryPrice,
            orderType: 'LIMIT',
            confidence: this.trendState.confirmation,
            notes: `Trend following entry: ${this.trendState.direction} trend`,
            metadata: {
                trendStrength: this.trendState.strength,
                stopLoss,
                takeProfit,
                indicators: this.indicators
            }
        };
    }

    private generateExitSignal(marketData: MarketData): StrategySignal {
        const position = this.positions.get(marketData.tokenAddress)!;
        
        return {
            action: 'SELL',
            tokenAddress: marketData.tokenAddress,
            amount: position.size,
            price: marketData.price,
            orderType: 'MARKET',
            confidence: this.trendState.confirmation,
            notes: 'Trend following exit',
            metadata: {
                exitReason: this.getExitReason(marketData, position),
                profitLoss: (marketData.price - position.entryPrice) * position.size,
                indicators: this.indicators
            }
        };
    }

    private calculatePositionSize(marketData: MarketData): number {
        const baseSize = this.config.maxPositionSize;
        const adjustment = this.trendState.strength * this.config.positionSizeAdjustment;
        return baseSize * (1 + adjustment);
    }

    private getExitReason(
        marketData: MarketData,
        position: any
    ): string {
        if (marketData.price <= position.stopLoss) return 'Stop loss';
        if (marketData.price >= position.takeProfit) return 'Take profit';
        return 'Trend reversal';
    }

    updateState(updateParams: any): void {
        super.updateState(updateParams);
        
        // Update trailing stops for open positions
        for (const [tokenAddress, position] of this.positions.entries()) {
            const currentPrice = updateParams.prices?.[tokenAddress];
            if (currentPrice && currentPrice > position.entryPrice) {
                const newStopLoss = currentPrice * (1 - this.config.trailingStopDistance / 100);
                if (newStopLoss > position.stopLoss) {
                    position.stopLoss = newStopLoss;
                }
            }
        }
    }
} 