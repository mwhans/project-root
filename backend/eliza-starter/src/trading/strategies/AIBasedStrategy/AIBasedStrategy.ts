import { BaseStrategy } from '../BaseStrategy';
import { MarketData, StrategySignal } from '../types';
import { 
    AIModelConfig, 
    ModelType, 
    FeatureSet,
    ModelPrediction,
    ModelMetrics 
} from './types';
import {
    loadModel,
    prepareFeatures,
    getPrediction,
    calculateModelMetrics
} from './utils';

export class AIBasedStrategy extends BaseStrategy {
    private config: AIModelConfig;
    private models: Map<ModelType, any>;
    private predictions: ModelPrediction[];
    private metrics: ModelMetrics;
    private lastUpdate: number;
    private positions: Map<string, {
        entryPrice: number;
        size: number;
        prediction: ModelPrediction;
        stopLoss: number;
        takeProfit: number;
    }>;

    constructor(config: Partial<AIModelConfig> = {}) {
        super('AIBased');
        
        this.config = {
            modelType: 'Ensemble',
            modelPath: './models',
            updateInterval: 60000,  // 1 minute
            featureLookback: 100,
            technicalFeatures: ['RSI', 'MACD', 'BBANDS'],
            marketFeatures: ['volume', 'liquidity', 'spread'],
            confidenceThreshold: 0.7,
            minProbability: 0.6,
            ensembleWeights: {
                LSTM: 0.4,
                RandomForest: 0.3,
                NeuralNetwork: 0.3,
                Ensemble: 0
            },
            maxPositionSize: 1000,
            positionSizeScaler: 0.5,
            stopLossPercent: 2,
            takeProfitPercent: 6,
            maxDrawdown: 10,
            performanceMetrics: {
                minAccuracy: 0.55,
                minPrecision: 0.6,
                maxDrawdown: 15
            },
            ...config
        };

        this.models = new Map();
        this.predictions = [];
        this.metrics = {} as ModelMetrics;
        this.lastUpdate = 0;
        this.positions = new Map();
        
        this.initialize();
    }

    private async initialize(): Promise<void> {
        try {
            if (this.config.modelType === 'Ensemble') {
                for (const [modelType, weight] of Object.entries(
                    this.config.ensembleWeights!
                )) {
                    if (weight > 0) {
                        const model = await loadModel(
                            modelType as ModelType,
                            `${this.config.modelPath}/${modelType.toLowerCase()}`
                        );
                        this.models.set(modelType as ModelType, model);
                    }
                }
            } else {
                const model = await loadModel(
                    this.config.modelType,
                    this.config.modelPath
                );
                this.models.set(this.config.modelType, model);
            }
        } catch (error) {
            console.error('Failed to initialize AI models:', error);
            throw error;
        }
    }

    async analyzeMarket(marketData: MarketData): Promise<StrategySignal[]> {
        const signals: StrategySignal[] = [];
        
        // Check if models need updating
        if (this.shouldUpdateModels()) {
            await this.updateModels();
        }

        // Prepare features
        const features = prepareFeatures(marketData, this.config);
        
        // Get predictions from all active models
        const predictions = await this.getPredictions(features);
        
        // Generate trading signals based on predictions
        if (this.validatePredictions(predictions)) {
            const signal = this.generateTradeSignal(
                predictions,
                marketData
            );
            if (signal) signals.push(signal);
        }

        // Manage existing positions
        this.positions.forEach((position, tokenAddress) => {
            const exitSignal = this.checkExitConditions(
                position,
                marketData
            );
            if (exitSignal) signals.push(exitSignal);
        });

        return signals;
    }

    private async getPredictions(
        features: FeatureSet
    ): Promise<ModelPrediction[]> {
        const predictions: ModelPrediction[] = [];

        for (const [modelType, model] of this.models.entries()) {
            try {
                const prediction = await getPrediction(
                    model,
                    features,
                    modelType
                );
                predictions.push(prediction);
            } catch (error) {
                console.error(`Prediction failed for ${modelType}:`, error);
            }
        }

        return predictions;
    }

    private validatePredictions(
        predictions: ModelPrediction[]
    ): boolean {
        if (predictions.length === 0) return false;

        // Check confidence and probability thresholds
        const avgConfidence = predictions.reduce(
            (sum, p) => sum + p.confidence,
            0
        ) / predictions.length;

        const avgProbability = predictions.reduce(
            (sum, p) => sum + p.probability,
            0
        ) / predictions.length;

        return (
            avgConfidence >= this.config.confidenceThreshold &&
            avgProbability >= this.config.minProbability &&
            this.metrics.accuracy >= this.config.performanceMetrics.minAccuracy
        );
    }

    private generateTradeSignal(
        predictions: ModelPrediction[],
        marketData: MarketData
    ): StrategySignal | null {
        const consensusDirection = this.getConsensusDirection(predictions);
        if (consensusDirection === 'NEUTRAL') return null;

        const positionSize = this.calculatePositionSize(
            predictions,
            marketData.price
        );

        const signal: StrategySignal = {
            action: consensusDirection === 'UP' ? 'BUY' : 'SELL',
            tokenAddress: marketData.tokenAddress,
            amount: positionSize,
            price: marketData.price,
            orderType: 'LIMIT',
            confidence: this.calculateAggregateConfidence(predictions),
            notes: `AI model consensus: ${consensusDirection}`,
            metadata: {
                predictions: predictions.map(p => ({
                    direction: p.direction,
                    confidence: p.confidence,
                    probability: p.probability
                })),
                metrics: this.metrics
            }
        };

        return signal;
    }

    private getConsensusDirection(
        predictions: ModelPrediction[]
    ): 'UP' | 'DOWN' | 'NEUTRAL' {
        const directions = predictions.map(p => p.direction);
        const upCount = directions.filter(d => d === 'UP').length;
        const downCount = directions.filter(d => d === 'DOWN').length;

        if (upCount > predictions.length / 2) return 'UP';
        if (downCount > predictions.length / 2) return 'DOWN';
        return 'NEUTRAL';
    }

    private calculatePositionSize(
        predictions: ModelPrediction[],
        currentPrice: number
    ): number {
        const baseSize = this.config.maxPositionSize / currentPrice;
        const confidence = this.calculateAggregateConfidence(predictions);
        return baseSize * confidence * this.config.positionSizeScaler;
    }

    private calculateAggregateConfidence(
        predictions: ModelPrediction[]
    ): number {
        return predictions.reduce(
            (sum, p) => sum + p.confidence * p.probability,
            0
        ) / predictions.length;
    }

    private async updateModels(): Promise<void> {
        this.lastUpdate = Date.now();
        
        // Update model metrics
        this.metrics = calculateModelMetrics(
            this.predictions,
            [] // actual outcomes
        );

        // Potentially retrain models if performance degrades
        if (this.shouldRetrainModels()) {
            await this.retrainModels();
        }
    }

    private shouldUpdateModels(): boolean {
        return Date.now() - this.lastUpdate >= this.config.updateInterval;
    }

    private shouldRetrainModels(): boolean {
        return (
            this.metrics.accuracy < this.config.performanceMetrics.minAccuracy ||
            this.metrics.precision < this.config.performanceMetrics.minPrecision ||
            this.metrics.drawdown > this.config.performanceMetrics.maxDrawdown
        );
    }

    private async retrainModels(): Promise<void> {
        // Implement model retraining logic
    }

    updateState(updateParams: any): void {
        super.updateState(updateParams);
        
        // Update prediction history
        if (updateParams.predictions) {
            this.predictions = [
                ...this.predictions,
                ...updateParams.predictions
            ].slice(-1000); // Keep last 1000 predictions
        }

        // Update position management
        if (updateParams.closedPositions) {
            updateParams.closedPositions.forEach((position: any) => {
                this.positions.delete(position.tokenAddress);
            });
        }
    }
} 