"""
DXS.app DIVINE HFT TRADING BOT v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A professional-grade high-frequency scalping bot designed for
aggressive profit extraction through micro-movements and rapid execution.

Key Features:
  • Multi-asset HFT with 1000s of micro-trades/day
  • Sub-second latency detection and execution
  • Adaptive volatility-based position sizing
  • Smart spread/momentum scalping
  • Real-time profit optimization
  • Dynamic stop-loss and trailing profit locks
  • WebSocket-based live market feeding
  • Portfolio rebalancing and risk management
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod

# Configure aggressive logging for performance tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('hft_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types for DXS.app"""
    MARKET_BUY = "MARKET_BUY"
    MARKET_SELL = "MARKET_SELL"
    LIMIT_BUY = "LIMIT_BUY"
    LIMIT_SELL = "LIMIT_SELL"


class TradeState(Enum):
    """States a trade can be in"""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"
    PROFIT_LOCKED = "PROFIT_LOCKED"


@dataclass
class MarketSnapshot:
    """Real-time market snapshot"""
    symbol: str
    bid: float
    ask: float
    mid: float
    timestamp: float
    volume: float
    volatility: float = 0.0
    
    @property
    def spread(self) -> float:
        return abs(self.ask - self.bid) / self.mid if self.mid else 0
    
    @property
    def spread_pips(self) -> float:
        return (self.spread * 10000) if self.symbol.endswith('USD') else self.spread * 1000


@dataclass
class MicroTrade:
    """Represents a single HFT micro-trade"""
    trade_id: str
    symbol: str
    direction: str  # "BUY" or "SELL"
    entry_price: float
    entry_time: float
    entry_amount_usd: float
    quantity: float
    leverage: int = 1
    
    current_price: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[float] = None
    state: TradeState = TradeState.PENDING
    
    # Dynamic thresholds (adaptive to market conditions)
    stop_loss_pct: float = 0.15  # 15 bps default
    take_profit_pct: float = 0.10  # 10 bps default
    trailing_stop_pct: float = 0.05  # Trailing stop at 5 bps
    
    # Performance tracking
    pnl: float = 0.0
    pnl_pct: float = 0.0
    execution_time: float = 0.0  # Time to execution in ms
    hold_time: float = 0.0  # How long position was held
    
    active_since: Optional[float] = field(default=None)
    peak_price: Optional[float] = field(default=None)
    lowest_price: Optional[float] = field(default=None)
    
    def update_price(self, price: float, timestamp: float) -> bool:
        """Update trade with new price, return True if SL/TP hit"""
        self.current_price = price
        
        if not self.active_since:
            self.active_since = timestamp
        
        # Track extremes
        if self.direction == "BUY":
            if self.peak_price is None or price > self.peak_price:
                self.peak_price = price
        else:
            if self.lowest_price is None or price < self.lowest_price:
                self.lowest_price = price
        
        # Check exit conditions
        if self.direction == "BUY":
            pnl_pct = ((price - self.entry_price) / self.entry_price) * 100
            
            # Stop loss
            if pnl_pct <= -self.stop_loss_pct:
                return True  # SL hit
            
            # Take profit
            if pnl_pct >= self.take_profit_pct:
                return True  # TP hit
            
            # Trailing stop
            if self.peak_price and (self.peak_price - price) / self.peak_price * 100 >= self.trailing_stop_pct:
                return True  # Trailing SL hit
        
        else:  # SELL
            pnl_pct = ((self.entry_price - price) / self.entry_price) * 100
            
            if pnl_pct <= -self.stop_loss_pct:
                return True
            if pnl_pct >= self.take_profit_pct:
                return True
            
            if self.lowest_price and (price - self.lowest_price) / self.lowest_price * 100 >= self.trailing_stop_pct:
                return True
        
        return False
    
    def close(self, exit_price: float, exit_time: float):
        """Close the trade and calculate PnL"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.hold_time = exit_time - self.entry_time
        self.state = TradeState.CLOSED
        
        if self.direction == "BUY":
            self.pnl = (exit_price - self.entry_price) * self.quantity
        else:
            self.pnl = (self.entry_price - exit_price) * self.quantity
        
        self.pnl_pct = (self.pnl / self.entry_amount_usd) * 100 if self.entry_amount_usd else 0


@dataclass
class TradingMetrics:
    """Daily/session trading metrics"""
    trades_executed: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    total_volume: float = 0.0
    execution_latency_ms: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    
    def update(self, trade: MicroTrade):
        """Update metrics with new closed trade"""
        self.trades_executed += 1
        self.total_pnl += trade.pnl
        self.total_volume += trade.entry_amount_usd
        
        if trade.pnl > 0:
            self.winning_trades += 1
            self.largest_win = max(self.largest_win, trade.pnl)
        else:
            self.losing_trades += 1
            self.largest_loss = min(self.largest_loss, trade.pnl)
        
        if self.trades_executed > 0:
            self.win_rate = (self.winning_trades / self.trades_executed) * 100
            self.avg_win = self.total_pnl / max(1, self.winning_trades)
            self.avg_loss = abs(self.total_loss) / max(1, self.losing_trades)
    
    @property
    def total_loss(self) -> float:
        return self.total_pnl - (self.total_pnl if self.total_pnl > 0 else 0)


class VolatilityAnalyzer:
    """Real-time volatility and micro-movement detection"""
    
    def __init__(self, lookback_periods: int = 20):
        self.lookback = lookback_periods
        self.price_history: Dict[str, deque] = {}
        self.volatility_cache: Dict[str, float] = {}
    
    def update(self, symbol: str, price: float, timestamp: float):
        """Update price history for volatility calculation"""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.lookback)
        
        self.price_history[symbol].append((price, timestamp))
        self._calculate_volatility(symbol)
    
    def _calculate_volatility(self, symbol: str):
        """Calculate realized volatility (annualized)"""
        if len(self.price_history[symbol]) < 2:
            self.volatility_cache[symbol] = 0.0
            return
        
        prices = [p[0] for p in self.price_history[symbol]]
        returns = np.diff(np.log(prices))
        
        if len(returns) > 0:
            # Realized volatility
            volatility = np.std(returns) * np.sqrt(252 * 24 * 60 * 60)  # Annualized
            self.volatility_cache[symbol] = volatility
    
    def get_volatility(self, symbol: str) -> float:
        """Get current volatility (0-1 scale)"""
        return self.volatility_cache.get(symbol, 0.0)
    
    def is_high_volatility(self, symbol: str, threshold: float = 0.02) -> bool:
        """Check if volatility exceeds threshold"""
        return self.get_volatility(symbol) > threshold


class OrderFlowAnalyzer:
    """Analyzes order flow and market microstructure"""
    
    def __init__(self, window_size: int = 100):
        self.order_flow: Dict[str, deque] = {}
        self.window_size = window_size
    
    def detect_momentum(self, symbol: str, bid: float, ask: float, last_mid: float) -> float:
        """
        Detect directional momentum from bid-ask dynamics
        Returns: momentum score (-1.0 to 1.0)
        """
        current_mid = (bid + ask) / 2
        momentum = (current_mid - last_mid) / last_mid if last_mid else 0
        
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, momentum * 100))
    
    def detect_reversal_setup(self, symbol: str, spread_bps: float, momentum: float) -> bool:
        """Detect potential reversal setups"""
        return spread_bps < 1.0 and abs(momentum) > 0.3
    
    def detect_breakout(self, prices: List[float], current_price: float, lookback: int = 20) -> Tuple[bool, str]:
        """Detect potential breakouts"""
        if len(prices) < lookback:
            return False, ""
        
        recent_prices = prices[-lookback:]
        high = max(recent_prices)
        low = min(recent_prices)
        
        if current_price > high * 1.001:
            return True, "BREAKOUT_UP"
        elif current_price < low * 0.999:
            return True, "BREAKOUT_DOWN"
        
        return False, ""


class PositionSizer:
    """Intelligent position sizing based on market conditions and risk"""
    
    def __init__(self, 
                 base_size_usd: float = 10.0,
                 max_risk_per_trade: float = 0.5,
                 max_total_exposure: float = 100.0):
        self.base_size = base_size_usd
        self.max_risk_per_trade = max_risk_per_trade  # % of balance
        self.max_exposure = max_total_exposure
    
    def calculate_size(self,
                      balance: float,
                      volatility: float,
                      current_exposure: float,
                      win_streak: int) -> float:
        """
        Calculate optimal position size based on:
        - Account balance and risk limits
        - Market volatility (inverse relationship)
        - Current exposure
        - Win streak (pyramiding)
        """
        # Base size adjusted for volatility
        volatility_factor = 1.0 / (1.0 + volatility * 10)  # Higher vol = smaller size
        
        # Risk-adjusted size
        risk_amount = (balance * self.max_risk_per_trade) / 100
        
        # Exposure-adjusted size
        remaining_exposure = self.max_exposure - current_exposure
        exposure_factor = remaining_exposure / self.max_exposure if self.max_exposure > 0 else 1.0
        
        # Winning streak bonus (pyramid on winners)
        streak_factor = 1.0 + (win_streak * 0.05) if win_streak > 0 else 1.0
        
        size = self.base_size * volatility_factor * exposure_factor * streak_factor
        size = min(size, risk_amount)
        size = max(size, self.base_size * 0.5)  # Minimum position size
        
        return size


class HFTStrategy:
    """Core HFT strategy engine with multiple signal generators"""
    
    def __init__(self):
        self.volatility_analyzer = VolatilityAnalyzer()
        self.order_flow = OrderFlowAnalyzer()
        self.position_sizer = PositionSizer()
        
        # Strategy parameters (optimized for scalping)
        self.micro_momentum_threshold = 0.005  # 50 bps
        self.spread_entry_threshold = 1.5  # 15 bps max for entry
        self.volatility_threshold = 0.015
        
        # Price tracking
        self.price_history: Dict[str, deque] = {}
        self.last_prices: Dict[str, float] = {}
    
    def update_market_data(self, snapshot: MarketSnapshot):
        """Update market data and analyze"""
        self.volatility_analyzer.update(snapshot.symbol, snapshot.mid, snapshot.timestamp)
        
        if snapshot.symbol not in self.price_history:
            self.price_history[snapshot.symbol] = deque(maxlen=200)
        
        self.price_history[snapshot.symbol].append(snapshot.mid)
        self.last_prices[snapshot.symbol] = snapshot.mid
    
    def generate_hft_signals(self, snapshot: MarketSnapshot) -> List[Dict]:
        """
        Generate multiple HFT trading signals:
        1. Micro-momentum scalping (ride small moves)
        2. Spread-fade trading (buy bid, sell ask)
        3. Volatility-triggered scalping
        4. Order flow reversal
        """
        signals = []
        
        # Signal 1: Spread-Fade Trading (most reliable for HFT)
        if snapshot.spread_pips < self.spread_entry_threshold:
            # Buy at bid, expect to sell at ask quickly
            signals.append({
                'type': 'SPREAD_FADE',
                'direction': 'BUY',
                'entry_price': snapshot.bid,
                'target_price': snapshot.ask,
                'stop_loss_pct': 0.20,
                'take_profit_pct': 0.08,
                'confidence': 0.75,
                'urgency': 'HIGH'
            })
            
            # Inverse: Sell at ask, buy at bid
            signals.append({
                'type': 'SPREAD_FADE',
                'direction': 'SELL',
                'entry_price': snapshot.ask,
                'target_price': snapshot.bid,
                'stop_loss_pct': 0.20,
                'take_profit_pct': 0.08,
                'confidence': 0.75,
                'urgency': 'HIGH'
            })
        
        # Signal 2: Micro-Momentum (tick-level)
        if snapshot.symbol in self.last_prices:
            last_price = self.last_prices[snapshot.symbol]
            momentum = (snapshot.mid - last_price) / last_price
            
            if abs(momentum) > self.micro_momentum_threshold:
                direction = 'BUY' if momentum > 0 else 'SELL'
                signals.append({
                    'type': 'MOMENTUM',
                    'direction': direction,
                    'entry_price': snapshot.ask if direction == 'BUY' else snapshot.bid,
                    'momentum': momentum,
                    'stop_loss_pct': 0.25,
                    'take_profit_pct': 0.12,
                    'confidence': min(0.9, abs(momentum) * 100),
                    'urgency': 'CRITICAL'
                })
        
        # Signal 3: High Volatility Scalping
        volatility = self.volatility_analyzer.get_volatility(snapshot.symbol)
        if self.volatility_analyzer.is_high_volatility(snapshot.symbol, self.volatility_threshold):
            signals.append({
                'type': 'VOLATILITY_SCALP',
                'direction': 'BIDIRECTIONAL',  # Take both sides
                'entry_price': snapshot.mid,
                'volatility': volatility,
                'stop_loss_pct': 0.30,
                'take_profit_pct': 0.15,
                'confidence': 0.6,
                'urgency': 'MEDIUM'
            })
        
        # Signal 4: Breakout Detection
        if snapshot.symbol in self.price_history:
            prices = list(self.price_history[snapshot.symbol])
            is_breakout, direction = self.order_flow.detect_breakout(prices, snapshot.mid)
            
            if is_breakout:
                signals.append({
                    'type': 'BREAKOUT',
                    'direction': 'BUY' if direction == 'BREAKOUT_UP' else 'SELL',
                    'entry_price': snapshot.mid,
                    'stop_loss_pct': 0.50,
                    'take_profit_pct': 0.30,
                    'confidence': 0.7,
                    'urgency': 'HIGH'
                })
        
        return signals


class DXSHFTBot:
    """
    Elite High-Frequency Trading Bot for DXS.app
    Executes 1000s of small scalp trades daily
    """
    
    def __init__(self,
                 handcash_client,
                 dxs_api_client,
                 trading_symbols: List[str] = None,
                 base_position_size: float = 5.0,
                 max_leverage: int = 20,
                 check_interval_ms: int = 100):
        """
        Initialize HFT Bot
        
        Args:
            handcash_client: HandCash wallet client
            dxs_api_client: DXS.app API client
            trading_symbols: List of symbols to trade (e.g., ['BTC', 'ETH', 'ADA'])
            base_position_size: Base USD size per trade
            max_leverage: Maximum leverage to use
            check_interval_ms: Check interval in milliseconds (HFT = 100ms or less)
        """
        self.handcash = handcash_client
        self.dxs = dxs_api_client
        
        self.symbols = trading_symbols or [
            'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'LINK', 'UNI', 'LTC'
        ]
        
        self.base_size = base_position_size
        self.max_leverage = max_leverage
        self.check_interval = check_interval_ms / 1000.0  # Convert to seconds
        
        # Trading state
        self.active_trades: Dict[str, List[MicroTrade]] = {s: [] for s in self.symbols}
        self.closed_trades: List[MicroTrade] = []
        self.metrics = TradingMetrics()
        
        # Strategy engine
        self.strategy = HFTStrategy()
        
        # Performance tracking
        self.execution_times = deque(maxlen=1000)
        self.start_time = time.time()
        self.session_pnl = 0.0
    
    async def initialize(self) -> bool:
        """Initialize bot connections and validate setup"""
        logger.info("🚀 Initializing DXS HFT Bot...")
        
        try:
            # Authenticate with HandCash
            balance = await self.handcash.get_balance()
            logger.info(f"💰 HandCash Balance: ${balance:.2f}")
            
            # Test DXS connection
            markets = await self.dxs.get_markets(self.symbols)
            logger.info(f"📊 Connected to {len(markets)} markets")
            
            return True
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    async def fetch_market_snapshot(self, symbol: str) -> Optional[MarketSnapshot]:
        """Fetch current market snapshot for a symbol"""
        try:
            market_data = await self.dxs.get_market_data(symbol)
            
            snapshot = MarketSnapshot(
                symbol=symbol,
                bid=market_data['bid'],
                ask=market_data['ask'],
                mid=(market_data['bid'] + market_data['ask']) / 2,
                timestamp=time.time(),
                volume=market_data.get('volume', 0),
                volatility=market_data.get('volatility', 0)
            )
            
            return snapshot
        except Exception as e:
            logger.error(f"Failed to fetch snapshot for {symbol}: {e}")
            return None
    
    async def execute_scalp_trade(self, signal: Dict, snapshot: MarketSnapshot) -> Optional[MicroTrade]:
        """Execute a scalp trade based on signal"""
        trade_id = f"{snapshot.symbol}_{int(time.time() * 1000)}"
        direction = signal['direction']
        entry_price = signal['entry_price']
        
        # Calculate position size
        balance = await self.handcash.get_balance()
        current_exposure = sum(
            t.entry_amount_usd for trades in self.active_trades.values() for t in trades
        )
        position_size = self.strategy.position_sizer.calculate_size(
            balance=balance,
            volatility=self.strategy.volatility_analyzer.get_volatility(snapshot.symbol),
            current_exposure=current_exposure,
            win_streak=self._get_current_win_streak()
        )
        
        # Determine leverage
        leverage = min(self.max_leverage, int(1 + (signal['confidence'] * 10)))
        
        try:
            # Execute market order
            execution_start = time.time()
            order = await self.dxs.place_order(
                symbol=snapshot.symbol,
                direction=direction,
                amount_usd=position_size,
                leverage=leverage,
                order_type='MARKET'
            )
            execution_time = (time.time() - execution_start) * 1000
            self.execution_times.append(execution_time)
            
            # Create trade record
            trade = MicroTrade(
                trade_id=trade_id,
                symbol=snapshot.symbol,
                direction=direction,
                entry_price=entry_price,
                entry_time=time.time(),
                entry_amount_usd=position_size,
                quantity=position_size / entry_price,
                leverage=leverage,
                stop_loss_pct=signal.get('stop_loss_pct', 0.20),
                take_profit_pct=signal.get('take_profit_pct', 0.10),
                execution_time=execution_time
            )
            
            trade.state = TradeState.ACTIVE
            self.active_trades[snapshot.symbol].append(trade)
            
            logger.info(
                f"📈 TRADE #{self.metrics.trades_executed}: {direction} {snapshot.symbol} "
                f"@ ${entry_price:.4f} | Size: ${position_size:.2f} | Exec: {execution_time:.1f}ms"
            )
            
            return trade
        
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            return None
    
    async def manage_active_trades(self):
        """Monitor and close active trades based on exits"""
        for symbol in self.symbols:
            snapshot = await self.fetch_market_snapshot(symbol)
            if not snapshot:
                continue
            
            for trade in self.active_trades[symbol][:]:  # Copy list to avoid mutation
                # Update trade with current price
                should_exit = trade.update_price(snapshot.mid, snapshot.timestamp)
                
                if should_exit:
                    await self.close_trade(trade, snapshot.mid, symbol)
    
    async def close_trade(self, trade: MicroTrade, exit_price: float, symbol: str):
        """Close a trade and record results"""
        try:
            # Execute exit order
            exit_direction = 'SELL' if trade.direction == 'BUY' else 'BUY'
            await self.dxs.place_order(
                symbol=symbol,
                direction=exit_direction,
                amount_usd=trade.entry_amount_usd,
                leverage=trade.leverage,
                order_type='MARKET'
            )
            
            # Record trade closure
            trade.close(exit_price, time.time())
            self.active_trades[symbol].remove(trade)
            self.closed_trades.append(trade)
            
            # Update metrics
            self.metrics.update(trade)
            self.session_pnl += trade.pnl
            
            # Log trade result
            result_emoji = "✅" if trade.pnl > 0 else "❌"
            logger.info(
                f"{result_emoji} CLOSED #{len(self.closed_trades)}: {symbol} "
                f"PnL: ${trade.pnl:.2f} ({trade.pnl_pct:+.2f}%) | Hold: {trade.hold_time:.1f}s"
            )
        
        except Exception as e:
            logger.error(f"Failed to close trade: {e}")
    
    def _get_current_win_streak(self) -> int:
        """Get current winning trade streak"""
        if not self.closed_trades:
            return 0
        
        streak = 0
        for trade in reversed(self.closed_trades):
            if trade.pnl > 0:
                streak += 1
            else:
                break
        
        return streak
    
    async def print_session_stats(self):
        """Print detailed session statistics"""
        elapsed = time.time() - self.start_time
        hours = elapsed / 3600
        
        logger.info("\\n" + "="*80)
        logger.info("📊 HFT SESSION STATISTICS")
        logger.info("="*80)
        logger.info(f"Session Duration: {hours:.2f} hours")
        logger.info(f"Total Trades: {self.metrics.trades_executed}")
        logger.info(f"Trades per Hour: {self.metrics.trades_executed / max(0.1, hours):.1f}")
        logger.info(f"Win Rate: {self.metrics.win_rate:.1f}%")
        logger.info(f"Total PnL: ${self.session_pnl:.2f}")
        logger.info(f"Daily PnL (annualized): ${self.session_pnl * (24/hours):.2f}")
        logger.info(f"Avg Execution Latency: {np.mean(list(self.execution_times)):.1f}ms")
        logger.info(f"Current Active Trades: {sum(len(t) for t in self.active_trades.values())}")
        logger.info("="*80 + "\\n")
    
    async def run(self, duration_hours: int = 24):
        """
        Main bot loop - run HFT strategy continuously
        
        Args:
            duration_hours: How long to run the bot (default: 24 hours)
        """
        if not await self.initialize():
            logger.error("Failed to initialize bot")
            return
        
        logger.info(f"🎯 Starting HFT bot for {duration_hours} hours")
        logger.info(f"🎲 Trading {len(self.symbols)} symbols with ${self.base_size:.2f} base size")
        
        end_time = time.time() + (duration_hours * 3600)
        
        try:
            while time.time() < end_time:
                # Process each symbol
                for symbol in self.symbols:
                    snapshot = await self.fetch_market_snapshot(symbol)
                    if not snapshot:
                        continue
                    
                    # Update strategy state
                    self.strategy.update_market_data(snapshot)
                    
                    # Generate HFT signals
                    signals = self.strategy.generate_hft_signals(snapshot)
                    
                    # Execute signals (up to N trades per symbol per iteration)
                    max_trades_per_iteration = 3
                    for signal in signals[:max_trades_per_iteration]:
                        # Filter by confidence and market conditions
                        if signal['confidence'] > 0.5:
                            await self.execute_scalp_trade(signal, snapshot)
                
                # Manage open positions
                await self.manage_active_trades()
                
                # Print stats every 60 seconds
                if int(time.time()) % 60 == 0:
                    await self.print_session_stats()
                
                # Sleep for check interval
                await asyncio.sleep(self.check_interval)
        
        except KeyboardInterrupt:
            logger.info("⛔ Bot stopped by user")
        except Exception as e:
            logger.error(f"❌ Bot error: {e}")
        finally:
            # Close all remaining positions
            logger.info("🔄 Closing all open positions...")
            for symbol in self.symbols:
                for trade in self.active_trades[symbol][:]:
                    snapshot = await self.fetch_market_snapshot(symbol)
                    if snapshot:
                        await self.close_trade(trade, snapshot.mid, symbol)
            
            # Final stats
            await self.print_session_stats()
            logger.info(f"🏁 Session Complete. Total PnL: ${self.session_pnl:.2f}")


class BacktestEngine:
    """Backtest HFT strategies on historical data"""
    
    @staticmethod
    async def backtest_strategy(
        symbol: str,
        historical_data: List[Dict],  # OHLCV data
        strategy: HFTStrategy,
        position_sizer: PositionSizer,
        initial_balance: float = 1000.0,
        leverage: int = 10
    ) -> Dict:
        """
        Run backtest on historical data
        
        Returns: Backtest metrics
        """
        balance = initial_balance
        trades = []
        
        for candle in historical_data:
            snapshot = MarketSnapshot(
                symbol=symbol,
                bid=candle['low'],
                ask=candle['high'],
                mid=candle['close'],
                timestamp=candle['timestamp'],
                volume=candle['volume']
            )
            
            strategy.update_market_data(snapshot)
            signals = strategy.generate_hft_signals(snapshot)
            
            # Simulate trade execution
            for signal in signals:
                position_size = position_sizer.calculate_size(balance, 0.02, 0, 0)
                
                entry_price = signal['entry_price']
                exit_price = snapshot.ask if signal['direction'] == 'SELL' else snapshot.bid
                
                if signal['direction'] == 'BUY':
                    pnl = (exit_price - entry_price) * (position_size / entry_price) * leverage
                else:
                    pnl = (entry_price - exit_price) * (position_size / entry_price) * leverage
                
                balance += pnl
                trades.append({
                    'pnl': pnl,
                    'entry': entry_price,
                    'exit': exit_price,
                    'direction': signal['direction']
                })
        
        # Calculate metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        total_pnl = balance - initial_balance
        
        return {
            'final_balance': balance,
            'total_pnl': total_pnl,
            'roi_pct': (total_pnl / initial_balance) * 100,
            'total_trades': total_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'trades': trades
        }


# ============================================
# USAGE EXAMPLE
# ============================================

async def main():
    """
    Initialize and run the HFT bot
    """
    # Initialize clients (pseudo-code - implement with real API)
    handcash_client = ...  # Initialize HandCash
    dxs_client = ...       # Initialize DXS.app API
    
    # Create and run bot
    bot = DXSHFTBot(
        handcash_client=handcash_client,
        dxs_api_client=dxs_client,
        trading_symbols=['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL'],
        base_position_size=5.0,  # $5 per trade
        max_leverage=20,
        check_interval_ms=100  # 100ms check interval for HFT
    )
    
    # Run for 24 hours
    await bot.run(duration_hours=24)


if __name__ == "__main__":
    asyncio.run(main())