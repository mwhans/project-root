import { SolanaTradeManager, TradeParams } from './TradeManager.js';
import { TradeRAGProvider } from './TradeRAGProvider.js';
import { ITradingStrategy } from './strategies/types';
import { MACrossoverStrategy } from './strategies/MACrossoverStrategy';

export class RAGIntegratedTradeManager extends SolanaTradeManager {
  private ragProvider: TradeRAGProvider;
  private strategies: Map<string, ITradingStrategy>;

  constructor(connection: any /* or Connection */) {
    super(connection);
    this.ragProvider = new TradeRAGProvider();
    this.strategies = new Map();
    
    // Initialize default strategies
    this.addStrategy(new MACrossoverStrategy({
      minLiquidityUsd: 250000,
      positionSizeUsd: 100
    }));
  }

  addStrategy(strategy: ITradingStrategy): void {
    this.strategies.set(strategy.strategyName, strategy);
  }

  removeStrategy(strategyName: string): void {
    this.strategies.delete(strategyName);
  }

  async executeStrategyAnalysis(marketData: any): Promise<void> {
    for (const strategy of this.strategies.values()) {
      try {
        const signals = await strategy.analyzeMarket(marketData);
        if (signals.length && signals.some(s => s.action !== 'HOLD')) {
          await strategy.executeTrades(signals, this);
        }
      } catch (error) {
        console.error(`Strategy ${strategy.strategyName} failed:`, error);
      }
    }
  }

  /**
   * Overriding executeTrade to incorporate specialized AI-based RAG.
   */
  async executeTrade(params: TradeParams): Promise<string> {
    // 1) Retrieve specialized context from internal tables
    const userQuery = `User wants to trade from ${params.inputToken} to ${params.outputToken}. 
      Slippage: ${params.slippage}, amount: ${params.amount}`;
    const tradeInsights = await this.ragProvider.retrieveTradeInsights(userQuery);
    console.log('TradeRAG insights:', tradeInsights);

    // 2) Optionally do a further risk check or refine logic
    // e.g. if the insights indicate suspicious activity, bail out or adjust

    // 3) Call the normal SolanaTradeManager logic
    return super.executeTrade(params);
  }
} 