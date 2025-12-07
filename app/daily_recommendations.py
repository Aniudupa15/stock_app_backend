# from typing import List, Dict
# from app.predictor import StockPredictor
# from app.market_analyzer import MarketAnalyzer
# from datetime import datetime

# class DailyRecommendations:
#     """
#     Generate daily buy/sell recommendations based on comprehensive analysis
#     """
    
#     # Popular stocks to analyze
#     WATCHLIST_STOCKS = [
#         'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
#         'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK',
#         'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA',
#         'TITAN', 'BAJFINANCE', 'WIPRO', 'TATAMOTORS', 'HCLTECH'
#     ]
    
#     def __init__(self):
#         self.predictor = StockPredictor()
#         self.analyzer = MarketAnalyzer()
    
#     def get_daily_recommendations(
#         self, 
#         stocks: List[str] = None,
#         min_score: int = 3
#     ) -> Dict:
#         """
#         Get daily buy/sell recommendations
        
#         Args:
#             stocks: List of stocks to analyze (default: top 20)
#             min_score: Minimum score for recommendation (1-5)
            
#         Returns:
#             Categorized recommendations with detailed analysis
#         """
#         if stocks is None:
#             stocks = self.WATCHLIST_STOCKS
        
#         buy_recommendations = []
#         sell_recommendations = []
#         hold_recommendations = []
        
#         for ticker in stocks:
#             try:
#                 recommendation = self._analyze_stock(ticker)
                
#                 if recommendation:
#                     # Categorize based on signal and score
#                     if recommendation['overall_signal'] == 'STRONG_BUY':
#                         buy_recommendations.append(recommendation)
#                     elif recommendation['overall_signal'] == 'BUY' and recommendation['score'] >= min_score:
#                         buy_recommendations.append(recommendation)
#                     elif recommendation['overall_signal'] == 'STRONG_SELL':
#                         sell_recommendations.append(recommendation)
#                     elif recommendation['overall_signal'] == 'SELL' and recommendation['score'] <= -min_score:
#                         sell_recommendations.append(recommendation)
#                     else:
#                         hold_recommendations.append(recommendation)
            
#             except Exception as e:
#                 print(f"Error analyzing {ticker}: {e}")
#                 continue
        
#         # Sort by score
#         buy_recommendations.sort(key=lambda x: x['score'], reverse=True)
#         sell_recommendations.sort(key=lambda x: x['score'])
#         hold_recommendations.sort(key=lambda x: abs(x['score']))
        
#         return {
#             "generated_at": datetime.now().isoformat(),
#             "market_date": datetime.now().strftime("%Y-%m-%d"),
#             "total_analyzed": len(stocks),
#             "summary": {
#                 "strong_buys": len([r for r in buy_recommendations if r['overall_signal'] == 'STRONG_BUY']),
#                 "buys": len(buy_recommendations),
#                 "holds": len(hold_recommendations),
#                 "sells": len(sell_recommendations),
#                 "strong_sells": len([r for r in sell_recommendations if r['overall_signal'] == 'STRONG_SELL'])
#             },
#             "recommendations": {
#                 "buy": buy_recommendations[:10],  # Top 10 buys
#                 "sell": sell_recommendations[:10],  # Top 10 sells
#                 "hold": hold_recommendations[:5]    # Top 5 holds
#             }
#         }
    
#     def _analyze_stock(self, ticker: str) -> Dict:
#         """
#         Comprehensive analysis of a single stock
        
#         Returns:
#             Detailed recommendation with score
#         """
#         # Get ML prediction
#         prediction = self.predictor.predict(ticker)
#         if not prediction:
#             return None
        
#         # Get technical analysis
#         analysis = self.analyzer.get_stock_analysis(ticker, period="3mo")
#         if not analysis:
#             return None
        
#         # Calculate comprehensive score
#         score = 0
#         reasons = []
        
#         # 1. ML Prediction Score (±2 points)
#         if prediction['signal'] == 'BUY':
#             score += 2
#             reasons.append(f"ML predicts +{prediction['predicted_return_pct']}% return")
#         elif prediction['signal'] == 'SELL':
#             score -= 2
#             reasons.append(f"ML predicts {prediction['predicted_return_pct']}% return")
        
#         # 2. RSI Score (±1 point)
#         rsi = analysis['indicators']['rsi']
#         if rsi <= 30:
#             score += 1
#             reasons.append(f"RSI oversold ({rsi:.1f})")
#         elif rsi >= 70:
#             score -= 1
#             reasons.append(f"RSI overbought ({rsi:.1f})")
        
#         # 3. MACD Score (±1 point)
#         if analysis['indicators']['macd_trend'] == 'Bullish':
#             score += 1
#             reasons.append("MACD bullish")
#         else:
#             score -= 1
#             reasons.append("MACD bearish")
        
#         # 4. Moving Average Score (±1 point)
#         current_price = analysis['current_price']
#         sma_20 = analysis['moving_averages'].get('sma_20', 0)
#         sma_50 = analysis['moving_averages'].get('sma_50', 0)
        
#         if current_price > sma_20 > sma_50:
#             score += 1
#             reasons.append("Strong uptrend (Price > SMA20 > SMA50)")
#         elif current_price < sma_20 < sma_50:
#             score -= 1
#             reasons.append("Strong downtrend (Price < SMA20 < SMA50)")
        
#         # 5. Volume Score (±1 point)
#         volume_ratio = analysis['volume']['ratio']
#         if volume_ratio > 1.5:
#             score += 1 if score > 0 else -1
#             reasons.append(f"High volume ({volume_ratio:.1f}x average)")
        
#         # Determine overall signal
#         if score >= 4:
#             overall_signal = 'STRONG_BUY'
#             action = 'BUY'
#             confidence = 'High'
#         elif score >= 2:
#             overall_signal = 'BUY'
#             action = 'BUY'
#             confidence = 'Medium'
#         elif score <= -4:
#             overall_signal = 'STRONG_SELL'
#             action = 'SELL'
#             confidence = 'High'
#         elif score <= -2:
#             overall_signal = 'SELL'
#             action = 'SELL'
#             confidence = 'Medium'
#         else:
#             overall_signal = 'HOLD'
#             action = 'HOLD'
#             confidence = 'Low'
        
#         return {
#             "ticker": prediction['ticker'],
#             "current_price": current_price,
#             "overall_signal": overall_signal,
#             "action": action,
#             "confidence": confidence,
#             "score": score,
#             "reasons": reasons,
#             "ml_prediction": {
#                 "predicted_price": prediction['predicted_close'],
#                 "predicted_return": prediction['predicted_return_pct'],
#                 "signal": prediction['signal'],
#                 "entry": prediction['entry_price'],
#                 "target": prediction['target_price'],
#                 "stop_loss": prediction['stop_loss']
#             },
#             "technical_analysis": {
#                 "rsi": rsi,
#                 "rsi_signal": analysis['indicators']['rsi_signal'],
#                 "macd_trend": analysis['indicators']['macd_trend'],
#                 "trend": analysis['trend'],
#                 "recommendation": analysis['technical_signals']['recommendation']
#             },
#             "risk_reward": {
#                 "potential_gain": round(((prediction['target_price'] - current_price) / current_price) * 100, 2),
#                 "potential_loss": round(((prediction['stop_loss'] - current_price) / current_price) * 100, 2)
#             }
#         }
    
#     def get_top_picks(self, category: str = "buy", limit: int = 5) -> List[Dict]:
#         """
#         Get top stock picks for the day
        
#         Args:
#             category: "buy", "sell", or "momentum"
#             limit: Number of picks to return
            
#         Returns:
#             List of top recommendations
#         """
#         recommendations = self.get_daily_recommendations()
        
#         if category.lower() == "buy":
#             return recommendations['recommendations']['buy'][:limit]
#         elif category.lower() == "sell":
#             return recommendations['recommendations']['sell'][:limit]
#         elif category.lower() == "momentum":
#             # Get stocks with highest volume and positive score
#             all_recs = (recommendations['recommendations']['buy'] + 
#                        recommendations['recommendations']['hold'])
#             return sorted(all_recs, key=lambda x: x['score'], reverse=True)[:limit]
        
#         return []



from typing import List, Dict, Optional
from app.predictor import StockPredictor
from app.market_analyzer import MarketAnalyzer
from datetime import datetime
import time
import logging

logger = logging.getLogger("daily_recommendations")
logger.setLevel(logging.INFO)


class DailyRecommendations:
    """
    Generate daily buy/sell recommendations based on comprehensive analysis.

    Resilient to cloud fetch failures:
    - Retries data fetches
    - Attempts ticker variants (.NS, .NSE)
    - Defensive checks for missing keys in analysis/prediction
    """

    # Popular stocks to analyze (base tickers); yfinance often needs exchange suffix
    WATCHLIST_STOCKS = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
        'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK',
        'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA',
        'TITAN', 'BAJFINANCE', 'WIPRO', 'TATAMOTORS', 'HCLTECH'
    ]

    # Ticker suffixes to try when fetching from yfinance-like sources
    TICKER_SUFFIXES = ["", ".NS", ".NSE"]

    def __init__(self, max_retries: int = 3, retry_backoff: float = 1.0):
        self.predictor = StockPredictor()
        self.analyzer = MarketAnalyzer()
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    # ------------------------
    # Public API
    # ------------------------
    def get_daily_recommendations(
        self,
        stocks: Optional[List[str]] = None,
        min_score: int = 3
    ) -> Dict:
        """
        Get daily buy/sell recommendations

        Args:
            stocks: List of stocks to analyze (default: top 20)
            min_score: Minimum score for recommendation (1-5)

        Returns:
            Categorized recommendations with detailed analysis and error summary
        """
        if stocks is None:
            stocks = self.WATCHLIST_STOCKS

        buy_recommendations = []
        sell_recommendations = []
        hold_recommendations = []
        errors = []

        for ticker in stocks:
            try:
                recommendation = self._analyze_stock_with_fallbacks(ticker)
                if recommendation is None:
                    errors.append({"ticker": ticker, "reason": "no_data_or_error"})
                    continue

                # categorize based on computed fields
                sig = recommendation.get('overall_signal', 'HOLD')
                score = recommendation.get('score', 0)

                if sig == 'STRONG_BUY':
                    buy_recommendations.append(recommendation)
                elif sig == 'BUY' and score >= min_score:
                    buy_recommendations.append(recommendation)
                elif sig == 'STRONG_SELL':
                    sell_recommendations.append(recommendation)
                elif sig == 'SELL' and score <= -min_score:
                    sell_recommendations.append(recommendation)
                else:
                    hold_recommendations.append(recommendation)

            except Exception as e:
                logger.exception("Unhandled error analyzing %s: %s", ticker, e)
                errors.append({"ticker": ticker, "reason": str(e)})
                continue

        # Sort by score
        buy_recommendations.sort(key=lambda x: x.get('score', 0), reverse=True)
        sell_recommendations.sort(key=lambda x: x.get('score', 0))
        # For holds, sort by absolute score descending (stronger signal first)
        hold_recommendations.sort(key=lambda x: abs(x.get('score', 0)), reverse=True)

        result = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "market_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "total_analyzed": len(stocks),
            "summary": {
                "strong_buys": len([r for r in buy_recommendations if r.get('overall_signal') == 'STRONG_BUY']),
                "buys": len(buy_recommendations),
                "holds": len(hold_recommendations),
                "sells": len(sell_recommendations),
                "strong_sells": len([r for r in sell_recommendations if r.get('overall_signal') == 'STRONG_SELL']),
                "errors": len(errors)
            },
            "recommendations": {
                "buy": buy_recommendations[:10],
                "sell": sell_recommendations[:10],
                "hold": hold_recommendations[:5]
            },
            "errors": errors
        }

        return result

    def get_top_picks(self, category: str = "buy", limit: int = 5) -> List[Dict]:
        """
        Get top stock picks for the day

        Args:
            category: "buy", "sell", or "momentum"
            limit: Number of picks to return
        """
        recommendations = self.get_daily_recommendations()

        if category.lower() == "buy":
            return recommendations['recommendations']['buy'][:limit]
        elif category.lower() == "sell":
            return recommendations['recommendations']['sell'][:limit]
        elif category.lower() == "momentum":
            # Get stocks with highest volume and positive score (from buy + hold)
            all_recs = (recommendations['recommendations']['buy'] +
                        recommendations['recommendations']['hold'])
            # safe fallback if 'volume' or 'score' missing
            def momentum_key(x):
                vol = x.get("technical_analysis", {}).get("volume", {}).get("current", 0)
                score = x.get("score", 0)
                return (score, vol)

            return sorted(all_recs, key=momentum_key, reverse=True)[:limit]

        return []

    # ------------------------
    # Internal helpers
    # ------------------------
    def _analyze_stock_with_fallbacks(self, base_ticker: str) -> Optional[Dict]:
        """
        Attempt to fetch prediction + analysis trying multiple ticker suffixes and retries.
        Returns a fully-formed recommendation dict or None on failure.
        """
        last_error = None
        for suffix in self.TICKER_SUFFIXES:
            ticker_try = f"{base_ticker}{suffix}"
            # Try a few times per ticker variant
            for attempt in range(1, self.max_retries + 1):
                try:
                    # Try prediction
                    try:
                        prediction = self.predictor.predict(ticker_try)
                    except Exception as e_pred:
                        prediction = None
                        logger.debug("Predictor failed for %s (attempt %d): %s", ticker_try, attempt, e_pred)

                    if not prediction:
                        # small backoff before retrying
                        logger.debug("No prediction for %s (attempt %d). Retrying...", ticker_try, attempt)
                        time.sleep(self.retry_backoff * attempt)
                        continue

                    # Try analysis
                    try:
                        analysis = self.analyzer.get_stock_analysis(ticker_try, period="3mo")
                    except Exception as e_an:
                        analysis = None
                        logger.debug("Analyzer failed for %s (attempt %d): %s", ticker_try, attempt, e_an)

                    if not analysis:
                        logger.debug("No analysis for %s (attempt %d). Retrying...", ticker_try, attempt)
                        time.sleep(self.retry_backoff * attempt)
                        continue

                    # If both prediction and analysis are present, compute recommendation
                    rec = self._compute_recommendation(prediction, analysis)
                    # Attach resolved ticker used
                    if rec is not None:
                        rec.setdefault("resolved_ticker", ticker_try)
                    return rec

                except Exception as e:
                    last_error = e
                    logger.exception("Attempt %d failed for %s: %s", attempt, ticker_try, e)
                    time.sleep(self.retry_backoff * attempt)

        # All suffixes exhausted
        logger.warning("All ticker variants failed for %s. Last error: %s", base_ticker, last_error)
        return None

    def _compute_recommendation(self, prediction: Dict, analysis: Dict) -> Optional[Dict]:
        """
        Given valid prediction and analysis dicts, compute a recommendation dict.
        Defensive about missing keys.
        """
        try:
            # Defensive extraction with defaults
            pred_signal = (prediction.get('signal') or "").upper()
            predicted_return_pct = prediction.get('predicted_return_pct', 0)
            predicted_close = prediction.get('predicted_close', None)
            entry_price = prediction.get('entry_price', None)
            target_price = prediction.get('target_price', None)
            stop_loss = prediction.get('stop_loss', None)
            resolved_ticker = prediction.get('ticker') or prediction.get('symbol')

            # Indicators and values from analysis; use safe defaults
            indicators = analysis.get('indicators', {})
            rsi = float(indicators.get('rsi', 50.0))
            rsi_signal = indicators.get('rsi_signal', 'NEUTRAL')
            macd_trend = indicators.get('macd_trend', 'Neutral')
            moving_averages = analysis.get('moving_averages', {})
            sma_20 = float(moving_averages.get('sma_20') or 0.0)
            sma_50 = float(moving_averages.get('sma_50') or 0.0)
            current_price = float(analysis.get('current_price') or 0.0)
            volume_info = analysis.get('volume', {})
            volume_ratio = float(volume_info.get('ratio') or volume_info.get('vol_ratio') or 0.0)

            score = 0
            reasons = []

            # 1. ML Prediction Score (±2 points)
            if pred_signal == 'BUY':
                score += 2
                reasons.append(f"ML predicts +{predicted_return_pct}% return")
            elif pred_signal == 'SELL':
                score -= 2
                reasons.append(f"ML predicts {predicted_return_pct}% return")

            # 2. RSI Score (±1 point)
            try:
                if rsi <= 30:
                    score += 1
                    reasons.append(f"RSI oversold ({rsi:.1f})")
                elif rsi >= 70:
                    score -= 1
                    reasons.append(f"RSI overbought ({rsi:.1f})")
            except Exception:
                logger.debug("RSI parsing problem; rsi=%s", rsi)

            # 3. MACD Score (±1 point)
            if str(macd_trend).lower().startswith('bull'):
                score += 1
                reasons.append("MACD bullish")
            elif str(macd_trend).lower().startswith('bear'):
                score -= 1
                reasons.append("MACD bearish")

            # 4. Moving Average Score (±1 point)
            if current_price and sma_20 and sma_50:
                if current_price > sma_20 > sma_50:
                    score += 1
                    reasons.append("Strong uptrend (Price > SMA20 > SMA50)")
                elif current_price < sma_20 < sma_50:
                    score -= 1
                    reasons.append("Strong downtrend (Price < SMA20 < SMA50)")

            # 5. Volume Score (±1 point)
            try:
                if volume_ratio > 1.5:
                    # reward if trend positive, penalize if negative
                    score += 1 if score > 0 else -1
                    reasons.append(f"High volume ({volume_ratio:.1f}x average)")
            except Exception:
                logger.debug("Volume parsing problem; volume_ratio=%s", volume_ratio)

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

            # Safe risk/reward calculation (avoid division by zero)
            potential_gain = None
            potential_loss = None
            try:
                if current_price and target_price is not None:
                    potential_gain = round(((target_price - current_price) / current_price) * 100, 2)
                if current_price and stop_loss is not None:
                    potential_loss = round(((stop_loss - current_price) / current_price) * 100, 2)
            except Exception:
                logger.debug("Risk/reward calc failed for %s", resolved_ticker)

            recommendation = {
                "ticker": resolved_ticker or "UNKNOWN",
                "current_price": current_price,
                "overall_signal": overall_signal,
                "action": action,
                "confidence": confidence,
                "score": score,
                "reasons": reasons,
                "ml_prediction": {
                    "predicted_price": predicted_close,
                    "predicted_return": predicted_return_pct,
                    "signal": pred_signal,
                    "entry": entry_price,
                    "target": target_price,
                    "stop_loss": stop_loss
                },
                "technical_analysis": {
                    "rsi": rsi,
                    "rsi_signal": rsi_signal,
                    "macd_trend": macd_trend,
                    "trend": analysis.get('trend'),
                    "recommendation": analysis.get('technical_signals', {}).get('recommendation'),
                    # include some volume details if present
                    "volume": {
                        "ratio": volume_ratio,
                        "current": volume_info.get('current'),
                        "average": volume_info.get('average')
                    }
                },
                "risk_reward": {
                    "potential_gain": potential_gain,
                    "potential_loss": potential_loss
                }
            }

            return recommendation

        except Exception as e:
            logger.exception("Failed to compute recommendation: %s", e)
            return None
