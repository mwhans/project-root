import { ExchangeQuote, TokenPair, ArbitrageOpportunity } from './types';

export function findSimpleArbitrage(
    quotes: Map<string, ExchangeQuote>,
    minProfitUsd: number,
    minLiquidityUsd: number
): ArbitrageOpportunity | null {
    let bestBuy: ExchangeQuote | null = null;
    let bestSell: ExchangeQuote | null = null;
    let maxProfit = 0;

    const validQuotes = Array.from(quotes.values())
        .filter(q => q.liquidityUsd >= minLiquidityUsd);

    for (const buyQuote of validQuotes) {
        for (const sellQuote of validQuotes) {
            if (buyQuote.exchange === sellQuote.exchange) continue;

            const profit = sellQuote.price - buyQuote.price;
            const profitAfterFees = profit - (
                buyQuote.fees.taker + sellQuote.fees.taker
            ) * sellQuote.price;

            if (profitAfterFees > maxProfit && profitAfterFees >= minProfitUsd) {
                maxProfit = profitAfterFees;
                bestBuy = buyQuote;
                bestSell = sellQuote;
            }
        }
    }

    if (!bestBuy || !bestSell) return null;

    return {
        type: 'SIMPLE',
        path: [
            {
                exchange: bestBuy.exchange,
                token: 'TOKEN_ADDRESS',  // Set in strategy
                action: 'BUY',
                price: bestBuy.price,
                amount: calculateOptimalAmount(bestBuy, bestSell)
            },
            {
                exchange: bestSell.exchange,
                token: 'TOKEN_ADDRESS',  // Set in strategy
                action: 'SELL',
                price: bestSell.price,
                amount: calculateOptimalAmount(bestBuy, bestSell)
            }
        ],
        expectedProfitUsd: maxProfit,
        confidence: calculateConfidence(bestBuy, bestSell)
    };
}

export function findTriangularArbitrage(
    pairs: Map<string, Map<string, ExchangeQuote>>,
    startToken: string,
    minProfitUsd: number
): ArbitrageOpportunity | null {
    const paths = findTriangularPaths(pairs, startToken);
    let bestPath: ArbitrageOpportunity | null = null;
    let maxProfit = 0;

    for (const path of paths) {
        const { profit, trades } = calculatePathProfit(path, pairs);
        if (profit > maxProfit && profit >= minProfitUsd) {
            maxProfit = profit;
            bestPath = {
                type: 'TRIANGULAR',
                path: trades,
                expectedProfitUsd: profit,
                confidence: calculatePathConfidence(trades)
            };
        }
    }

    return bestPath;
}

function calculateOptimalAmount(
    buyQuote: ExchangeQuote,
    sellQuote: ExchangeQuote
): number {
    // Consider liquidity on both sides
    const maxBuyAmount = buyQuote.liquidityUsd / buyQuote.price;
    const maxSellAmount = sellQuote.liquidityUsd / sellQuote.price;
    return Math.min(maxBuyAmount, maxSellAmount) * 0.95; // 95% to account for slippage
}

function calculateConfidence(
    buyQuote: ExchangeQuote,
    sellQuote: ExchangeQuote
): number {
    // Factor in liquidity, price stability, etc.
    const liquidityScore = Math.min(
        buyQuote.liquidityUsd,
        sellQuote.liquidityUsd
    ) / 1000000; // Normalize to 0-1
    return Math.min(liquidityScore, 1);
}

// Helper functions for triangular arbitrage
function findTriangularPaths(
    pairs: Map<string, Map<string, ExchangeQuote>>,
    startToken: string
): TokenPair[][] {
    // Implementation of path finding algorithm
    // Returns array of valid triangular paths
    return [];
}

function calculatePathProfit(
    path: TokenPair[],
    pairs: Map<string, Map<string, ExchangeQuote>>
): { profit: number; trades: any[] } {
    // Calculate the profit for a given path
    return { profit: 0, trades: [] };
} 