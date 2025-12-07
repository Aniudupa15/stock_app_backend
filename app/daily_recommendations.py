from typing import List, Dict
from predictor import StockPredictor
from market_analyzer import MarketAnalyzer
from datetime import datetime

class DailyRecommendations:
    """
    Generate daily buy/sell recommendations based on comprehensive analysis
    """
    
    # Popular stocks to analyze
    WATCHLIST_STOCKS = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
        'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK',
        'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA',
        'TITAN', 'BAJFINANCE', 'WIPRO', 'TATAMOTORS', 'HCLTECH'
    ]
    
    def __init__(self):
        self.predictor = StockPredictor()
        self.analyzer = MarketAnalyzer()
    
    def get_daily_recommendations(
        self, 
        stocks: List[str] = None,
        min_score: int = 3
    ) -> Dict:
        """
        Get daily buy/sell recommendations
        
        Args:
            stocks: List of stocks to analyze (default: top 20)
            min_score: Minimum score for recommendation (1-5)
            
        Returns:
            Categorized recommendations with detailed analysis
        """
        if stocks is None:
            stocks = self.WATCHLIST_STOCKS
        
        buy_recommendations = []
        sell_recommendations = []
        hold_recommendations = []
        
        for ticker in stocks:
            try:
                recommendation = self._analyze_stock(ticker)
                
                if recommendation:
                    # Categorize based on signal and score
                    if recommendation['overall_signal'] == 'STRONG_BUY':
                        buy_recommendations.append(recommendation)
                    elif recommendation['overall_signal'] == 'BUY' and recommendation['score'] >= min_score:
                        buy_recommendations.append(recommendation)
                    elif recommendation['overall_signal'] == 'STRONG_SELL':
                        sell_recommendations.append(recommendation)
                    elif recommendation['overall_signal'] == 'SELL' and recommendation['score'] <= -min_score:
                        sell_recommendations.append(recommendation)
                    else:
                        hold_recommendations.append(recommendation)
            
            except Exception as e:
                print(f"Error analyzing {ticker}: {e}")
                continue
        
        # Sort by score
        buy_recommendations.sort(key=lambda x: x['score'], reverse=True)
        sell_recommendations.sort(key=lambda x: x['score'])
        hold_recommendations.sort(key=lambda x: abs(x['score']))
        
        return {
            "generated_at": datetime.now().isoformat(),
            "market_date": datetime.now().strftime("%Y-%m-%d"),
            "total_analyzed": len(stocks),
            "summary": {
                "strong_buys": len([r for r in buy_recommendations if r['overall_signal'] == 'STRONG_BUY']),
                "buys": len(buy_recommendations),
                "holds": len(hold_recommendations),
                "sells": len(sell_recommendations),
                "strong_sells": len([r for r in sell_recommendations if r['overall_signal'] == 'STRONG_SELL'])
            },
            "recommendations": {
                "buy": buy_recommendations[:10],  # Top 10 buys
                "sell": sell_recommendations[:10],  # Top 10 sells
                "hold": hold_recommendations[:5]    # Top 5 holds
            }
        }
    
    def _analyze_stock(self, ticker: str) -> Dict:
        """
        Comprehensive analysis of a single stock
        
        Returns:
            Detailed recommendation with score
        """
        # Get ML prediction
        prediction = self.predictor.predict(ticker)
        if not prediction:
            return None
        
        # Get technical analysis
        analysis = self.analyzer.get_stock_analysis(ticker, period="3mo")
        if not analysis:
            return None
        
        # Calculate comprehensive score
        score = 0
        reasons = []
        
        # 1. ML Prediction Score (±2 points)
        if prediction['signal'] == 'BUY':
            score += 2
            reasons.append(f"ML predicts +{prediction['predicted_return_pct']}% return")
        elif prediction['signal'] == 'SELL':
            score -= 2
            reasons.append(f"ML predicts {prediction['predicted_return_pct']}% return")
        
        # 2. RSI Score (±1 point)
        rsi = analysis['indicators']['rsi']
        if rsi <= 30:
            score += 1
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi >= 70:
            score -= 1
            reasons.append(f"RSI overbought ({rsi:.1f})")
        
        # 3. MACD Score (±1 point)
        if analysis['indicators']['macd_trend'] == 'Bullish':
            score += 1
            reasons.append("MACD bullish")
        else:
            score -= 1
            reasons.append("MACD bearish")
        
        # 4. Moving Average Score (±1 point)
        current_price = analysis['current_price']
        sma_20 = analysis['moving_averages'].get('sma_20', 0)
        sma_50 = analysis['moving_averages'].get('sma_50', 0)
        
        if current_price > sma_20 > sma_50:
            score += 1
            reasons.append("Strong uptrend (Price > SMA20 > SMA50)")
        elif current_price < sma_20 < sma_50:
            score -= 1
            reasons.append("Strong downtrend (Price < SMA20 < SMA50)")
        
        # 5. Volume Score (±1 point)
        volume_ratio = analysis['volume']['ratio']
        if volume_ratio > 1.5:
            score += 1 if score > 0 else -1
            reasons.append(f"High volume ({volume_ratio:.1f}x average)")
        
        # Determine overall signal
        if score >= 4:
            overall_signal = 'STRONG_BUY'
            action = 'BUY'
            confidence = 'High'
        elif score >= 2:
            overall_signal = 'BUY'
            action = 'BUY'
            confidence = 'Medium'
        elif score <= -4:
            overall_signal = 'STRONG_SELL'
            action = 'SELL'
            confidence = 'High'
        elif score <= -2:
            overall_signal = 'SELL'
            action = 'SELL'
            confidence = 'Medium'
        else:
            overall_signal = 'HOLD'
            action = 'HOLD'
            confidence = 'Low'
        
        return {
            "ticker": prediction['ticker'],
            "current_price": current_price,
            "overall_signal": overall_signal,
            "action": action,
            "confidence": confidence,
            "score": score,
            "reasons": reasons,
            "ml_prediction": {
                "predicted_price": prediction['predicted_close'],
                "predicted_return": prediction['predicted_return_pct'],
                "signal": prediction['signal'],
                "entry": prediction['entry_price'],
                "target": prediction['target_price'],
                "stop_loss": prediction['stop_loss']
            },
            "technical_analysis": {
                "rsi": rsi,
                "rsi_signal": analysis['indicators']['rsi_signal'],
                "macd_trend": analysis['indicators']['macd_trend'],
                "trend": analysis['trend'],
                "recommendation": analysis['technical_signals']['recommendation']
            },
            "risk_reward": {
                "potential_gain": round(((prediction['target_price'] - current_price) / current_price) * 100, 2),
                "potential_loss": round(((prediction['stop_loss'] - current_price) / current_price) * 100, 2)
            }
        }
    
    def get_top_picks(self, category: str = "buy", limit: int = 5) -> List[Dict]:
        """
        Get top stock picks for the day
        
        Args:
            category: "buy", "sell", or "momentum"
            limit: Number of picks to return
            
        Returns:
            List of top recommendations
        """
        recommendations = self.get_daily_recommendations()
        
        if category.lower() == "buy":
            return recommendations['recommendations']['buy'][:limit]
        elif category.lower() == "sell":
            return recommendations['recommendations']['sell'][:limit]
        elif category.lower() == "momentum":
            # Get stocks with highest volume and positive score
            all_recs = (recommendations['recommendations']['buy'] + 
                       recommendations['recommendations']['hold'])
            return sorted(all_recs, key=lambda x: x['score'], reverse=True)[:limit]
        
        return []