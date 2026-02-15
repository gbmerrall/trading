# Trading Strategies

## Status Key
- [x] Implemented
- [ ] Planned

## Implemented

- [x] **Consecutive days (Buy/Sell)**: Buy after N consecutive down days, sell after N consecutive
  up days. Class: `ConsecutiveDaysStrategy`.
- [x] **Moving average crossover**: Generates buy signals when a short-term moving average crosses
  above a long-term moving average, and sell signals on the reverse crossover.
  Class: `MovingAverageCrossoverStrategy`.
- [x] **Relative strength index (RSI)**: Uses the RSI oscillator to buy when the indicator drops
  below an oversold threshold and sell when it exceeds an overbought threshold.
  Class: `RSIStrategy`.
- [x] **MACD**: Tracks the relationship between two exponential moving averages to identify momentum
  changes via MACD/signal line crossovers. Class: `MACDStrategy`.
- [x] **Bollinger bands**: Capitalizes on mean reversion by buying on lower band touches and selling
  on upper band touches. Class: `BollingerBandsStrategy`.
- [x] **Parabolic SAR**: Uses a trailing stop-loss system to determine trend direction and potential
  reversals based on price acceleration. Class: `ParabolicSARStrategy`.

## Planned: Technical Indicator Strategies

None remaining - all technical indicator strategies completed!

## Implemented: Price Action Strategies

- [x] **Simple breakout**: Enters a trade when the price exceeds an N-day high or low, anticipating
  trend continuation. Class: `BreakoutStrategy`.
- [x] **Gaps**: Exploits overnight price discontinuities, trading the gap fill or continuation.
  Class: `GapStrategy` (requires Open column).
- [x] **Fibonacci retracements**: Identifies potential support and resistance levels by measuring
  retracement of a prior price move at Fibonacci ratios (38.2%, 50%, 61.8%).
  Class: `FibonacciRetracementStrategy` (requires High/Low columns).

## Planned: Price Action Strategies

None remaining - all price action strategies completed!

## Implemented: Composite Strategies

- [x] **Mean reversion**: Combines RSI and Bollinger Bands to identify overbought/oversold
  conditions and trade the reversion to the mean. Class: `MeanReversionStrategy`.
- [x] **Momentum (ROC)**: Uses Rate-of-Change momentum indicator to buy when momentum exceeds
  positive threshold and sell when below negative threshold. Class: `MomentumStrategy`.
- [x] **Volatility (ATR)**: Uses Average True Range for breakout detection during high volatility
  periods. Class: `VolatilityStrategy` (requires High/Low columns).
- [x] **Ensemble/Combination**: A wrapper strategy that accepts multiple sub-strategies and
  aggregates their signals via majority voting. Class: `EnsembleStrategy`.

## Planned: Composite Strategies

None remaining - all composite strategies completed!

## Scope Constraints

- All strategies operate on **single-asset OHLCV data**. Cross-asset strategies (pairs trading,
  relative momentum) are out of scope until the runner supports multi-asset event loops.
- **Volatility strategies** use ATR from price data only. Options-based or VIX-based approaches
  require a different data model and are out of scope.
- **Momentum strategies** use single-asset Rate-of-Change. Cross-sectional momentum (ranking
  multiple assets) requires runner changes and is out of scope.
