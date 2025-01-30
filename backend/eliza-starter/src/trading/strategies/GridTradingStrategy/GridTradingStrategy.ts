import { BaseStrategy } from '../BaseStrategy';
import { MarketData, StrategySignal } from '../types';
import { 
    GridConfig, 
    GridState, 
    GridLevel 
} from './types';
import {
    calculateGridLevels,
    shouldRebalanceGrid,
    calculateGridProfitability
} from './utils';

export class GridTradingStrategy extends BaseStrategy {
    private config: GridConfig;
    private state: GridState;

    constructor(config: Partial<GridConfig> = {}) {
        super('GridTrading');
        
        this.config = {
            gridLevels: 5,
            gridSpacing: 0.01,
            dynamicSpacing: true,
            baseOrderSize: 100,
            sizeMultiplier: 1.2,
            maxTotalPositionSize: 10000,
            maxLevelsPerSide: 5,
            stopLossLevel: 0.15,    // 15% below reference
            maxDrawdown: 10,
            rebalanceThreshold: 0.05,
            minProfitPerTrade: 0.002,
            orderTimeout: 30000,
            rebalanceInterval: 3600000, // 1 hour
            ...config
        };

        this.state = {
            referencePrice: 0,
            levels: [],
            totalInvestment: 0,
            currentProfit: 0,
            lastRebalance: 0,
            activeOrders: new Map()
        };
    }

    async analyzeMarket(marketData: MarketData): Promise<StrategySignal[]> {
        const signals: StrategySignal[] = [];
        
        // Initialize or rebalance grid if needed
        if (
            this.state.levels.length === 0 ||
            shouldRebalanceGrid(
                marketData.price,
                this.state,
                this.config
            )
        ) {
            await this.rebalanceGrid(marketData);
        }

        // Check for grid level triggers
        this.state.levels
            .filter(level => level.status === 'ACTIVE')
            .forEach(level => {
                if (this.isLevelTriggered(level, marketData.price)) {
                    signals.push(
                        this.createGridSignal(level, marketData)
                    );
                }
            });

        // Check stop loss
        if (this.shouldTriggerStopLoss(marketData.price)) {
            signals.push(this.createEmergencyExitSignal(marketData));
        }

        return signals;
    }

    private async rebalanceGrid(marketData: MarketData): Promise<void> {
        // Cancel existing orders
        await this.cancelExistingOrders();

        // Calculate new grid levels
        this.state.referencePrice = marketData.price;
        this.state.levels = calculateGridLevels(
            marketData.price,
            this.config,
            marketData.volatility24h
        );

        // Update state
        this.state.lastRebalance = Date.now();
        this.updateGridState();
    }

    private isLevelTriggered(
        level: GridLevel,
        currentPrice: number
    ): boolean {
        return level.side === 'BUY' ?
            currentPrice <= level.price :
            currentPrice >= level.price;
    }

    private createGridSignal(
        level: GridLevel,
        marketData: MarketData
    ): StrategySignal {
        return {
            action: level.side,
            tokenAddress: marketData.tokenAddress,
            amount: level.size,
            price: level.price,
            orderType: 'LIMIT',
            confidence: 1,
            notes: `Grid ${level.side} at ${level.price}`,
            metadata: {
                gridLevel: level,
                profitability: calculateGridProfitability(
                    this.state,
                    marketData.price
                )
            }
        };
    }

    private createEmergencyExitSignal(
        marketData: MarketData
    ): StrategySignal {
        return {
            action: 'SELL',
            tokenAddress: marketData.tokenAddress,
            amount: this.calculateTotalPosition(),
            price: marketData.price,
            orderType: 'MARKET',
            confidence: 1,
            notes: 'Emergency grid stop loss',
            metadata: {
                reason: 'STOP_LOSS',
                profitability: calculateGridProfitability(
                    this.state,
                    marketData.price
                )
            }
        };
    }

    private shouldTriggerStopLoss(currentPrice: number): boolean {
        const deviation = (currentPrice - this.state.referencePrice) /
                         this.state.referencePrice;
        return Math.abs(deviation) > this.config.stopLossLevel;
    }

    private calculateTotalPosition(): number {
        return this.state.levels
            .filter(level => level.status === 'FILLED')
            .reduce((total, level) => total + level.size, 0);
    }

    async executeTrades(
        signals: StrategySignal[],
        tradeManager: any
    ): Promise<void> {
        for (const signal of signals) {
            try {
                const result = await tradeManager.executeTrade({
                    inputToken: signal.action === 'BUY' ? 'USDC' : signal.tokenAddress,
                    outputToken: signal.action === 'BUY' ? signal.tokenAddress : 'USDC',
                    amount: signal.amount,
                    price: signal.price,
                    orderType: signal.orderType,
                    timeout: this.config.orderTimeout
                });

                this.updateLevelStatus(signal, result);
            } catch (error) {
                console.error('Grid trade execution failed:', error);
                // Handle failed execution
                this.handleFailedExecution(signal);
            }
        }
    }

    private updateLevelStatus(
        signal: StrategySignal,
        result: any
    ): void {
        const level = signal.metadata?.gridLevel;
        if (level) {
            level.status = 'FILLED';
            level.orderId = result.orderId;
            this.updateGridState();
        }
    }

    private handleFailedExecution(signal: StrategySignal): void {
        const level = signal.metadata?.gridLevel;
        if (level) {
            level.status = 'CANCELLED';
            this.updateGridState();
        }
    }

    private async cancelExistingOrders(): Promise<void> {
        // Implementation depends on your trade manager
        for (const [orderId, level] of this.state.activeOrders) {
            try {
                // await tradeManager.cancelOrder(orderId);
                level.status = 'CANCELLED';
            } catch (error) {
                console.error('Failed to cancel order:', orderId, error);
            }
        }
        this.state.activeOrders.clear();
    }

    private updateGridState(): void {
        // Update active orders map
        this.state.activeOrders.clear();
        this.state.levels
            .filter(level => level.status === 'ACTIVE' && level.orderId)
            .forEach(level => {
                this.state.activeOrders.set(level.orderId!, level);
            });

        // Calculate total investment
        this.state.totalInvestment = this.state.levels
            .filter(level => level.status === 'FILLED' && level.side === 'BUY')
            .reduce((total, level) => total + (level.price * level.size), 0);
    }

    updateState(updateParams: any): void {
        super.updateState(updateParams);
        
        if (updateParams.filledOrders) {
            updateParams.filledOrders.forEach((order: any) => {
                const level = this.state.activeOrders.get(order.orderId);
                if (level) {
                    level.status = 'FILLED';
                    this.updateGridState();
                }
            });
        }
    }
} 