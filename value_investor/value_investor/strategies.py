import numpy as np
import pandas as pd
from typing import Dict
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from value_investor.preprocessing import create_sequences
from value_investor.config import (
    WINDOW_SIZE, HORIZON, BATCH_SIZE, NUM_UNITS, 
    EPOCHS, DROPOUT_PROP, LEARNING_RATE,
    BOLLINGER_WINDOW, VOLUME_THRESHOLD, STD_DEV,
    MIN_RETURN
)
from value_investor.logging import logger, metrics_logger

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.strategy_name = name
        logger.info(f"Initializing {self.strategy_name}")
    
    @abstractmethod
    def calculate_signals(self, company_dict: Dict) -> pd.DataFrame:
        """Generate trading signals - must be implemented by child classes"""
        pass
    
    def calculate_returns(self, df: pd.DataFrame, initial_cash: float = 100000) -> Dict:
        """Calculate investment returns based on signals"""
        logger.info(f"{self.strategy_name}: Calculating returns with initial_cash={initial_cash}")
        
        try:
            position = 0
            cash = initial_cash
            shares = 0
            trades = []
            peak_value = initial_cash
            max_drawdown = 0
            
            for idx, row in df.iterrows():
                current_value = cash if position == 0 else shares * row['Price']
                
                # Track drawdown
                if current_value > peak_value:
                    peak_value = current_value
                drawdown = (peak_value - current_value) / peak_value * 100
                max_drawdown = max(max_drawdown, drawdown)
                
                # Execute trades
                if row['Signal'] == 'BUY' and position == 0:
                    shares = cash / row['Price']
                    position = 1
                    trades.append({
                        'date': idx,
                        'type': 'BUY',
                        'price': row['Price'],
                        'shares': shares,
                        'value': cash
                    })
                    
                elif row['Signal'] == 'SELL' and position == 1:
                    cash = shares * row['Price']
                    trade_return = (cash - trades[-1]['value']) / trades[-1]['value'] * 100
                    trades.append({
                        'date': idx,
                        'type': 'SELL',
                        'price': row['Price'],
                        'shares': shares,
                        'value': cash,
                        'return': trade_return
                    })
                    shares = 0
                    position = 0
            
            # Calculate final results
            final_value = cash if position == 0 else shares * df['Price'].iloc[-1]
            total_return = (final_value - initial_cash) / initial_cash * 100
            num_trades = len(trades) // 2
            
            # Calculate trade statistics if we made any trades
            if num_trades > 0:
                trade_returns = [t['return'] for t in trades if t['type'] == 'SELL']
                win_rate = (sum(1 for r in trade_returns if r > 0) / len(trade_returns)) * 100
                avg_return = sum(trade_returns) / len(trade_returns)
            else:
                win_rate = 0
                avg_return = 0
            
            results = {
                'total_return': total_return,
                'number_of_trades': num_trades,
                'win_rate': win_rate,
                'avg_trade_return': avg_return,
                'max_drawdown': max_drawdown,
                'final_value': final_value
            }
            
            logger.info(f"{self.strategy_name}: Final return {total_return:.2f}%")
            return results
            
        except Exception as e:
            logger.error(f"{self.strategy_name}: Error calculating returns", exc_info=True)
            raise

    def analyze(self, company_dict: Dict, initial_investment: float = 100000) -> Dict:
        """Analyze a stock with the strategy"""
        logger.info(f"{self.strategy_name}: Starting analysis")
        
        try:
            # Get signals for test data
            test_data_with_signals = self.calculate_signals(company_dict)
            
            # Calculate returns
            returns = self.calculate_returns(test_data_with_signals, initial_investment)
            
            # Calculate risk metrics
            daily_returns = test_data_with_signals['Price'].pct_change()
            volatility = daily_returns.std() * np.sqrt(252) * 100
            sharpe_ratio = returns['total_return'] / (volatility if volatility != 0 else 1)
            
            analysis = {
                'returns': returns,
                'metrics': {
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"{self.strategy_name}: Error during analysis", exc_info=True)
            raise

class BollingerBandStrategy(BaseStrategy):
    """Trading strategy based on Bollinger Bands"""
    
    def __init__(self):
        super().__init__("Bollinger Bands")
    
    def calculate_signals(self, company_dict: Dict) -> pd.DataFrame:
        """Generate trading signals based on Bollinger Bands"""
        logger.info("Calculating Bollinger Bands signals")
        
        try:
            # Combine train and test data for continuous calculation
            full_data = pd.concat([company_dict['train'], company_dict['test']])
            
            # Calculate Bollinger Bands on full dataset
            full_data['middle_band'] = full_data['Price'].rolling(window=BOLLINGER_WINDOW).mean()
            std = full_data['Price'].rolling(window=BOLLINGER_WINDOW).std()
            full_data['upper_band'] = full_data['middle_band'] + STD_DEV * std
            full_data['lower_band'] = full_data['middle_band'] - STD_DEV * std
            
            # Volume confirmation on full dataset
            full_data['volume_ratio'] = full_data['Vol.'] / full_data['Vol.'].rolling(BOLLINGER_WINDOW).mean()
            
            # Extract just the test portion for signal generation
            test_data = full_data[company_dict['test'].index[0]:]
            df = test_data.copy()
            df['Signal'] = 'HOLD'
            
            # Generate signals only for test data
            position_open = False
            
            for idx, row in df.iterrows():
                if not position_open:
                    if (row['Price'] < row['lower_band'] and 
                        row['volume_ratio'] > VOLUME_THRESHOLD):
                        df.loc[idx, 'Signal'] = 'BUY'
                        position_open = True
                else:
                    if (row['Price'] > row['upper_band'] and 
                        row['volume_ratio'] > VOLUME_THRESHOLD):
                        df.loc[idx, 'Signal'] = 'SELL'
                        position_open = False
            
            logger.info(f"Generated signals: {df['Signal'].value_counts().to_dict()}")
            return df
            
        except Exception as e:
            logger.error("Error calculating Bollinger Bands signals", exc_info=True)
            raise

class LSTMStrategy(BaseStrategy):
    """LSTM strategy that predicts price direction for next HORIZON days"""
    
    def __init__(self):
        super().__init__("LSTM")
        self.model = self._build_model()
    
    def _build_model(self) -> Sequential:
        """Build simple LSTM model for price prediction"""
        try:
            model = Sequential([
                Input(shape=(WINDOW_SIZE, 1), name='input'),
                LSTM(units=NUM_UNITS, return_sequences=False, name='lstm'),
                Dropout(DROPOUT_PROP, name='dropout1'),
                Dense(units=HORIZON, name='output')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=LEARNING_RATE),
                loss='mse'
            )
            
            return model
            
        except Exception as e:
            logger.error("Error building LSTM model", exc_info=True)
            raise

    def train(self, df: pd.DataFrame, scaler: MinMaxScaler):
        """Train LSTM model on training data"""
        try:
            X, y = create_sequences(
                scaler.transform(df['Price'].values.reshape(-1, 1)),
                window_size=WINDOW_SIZE,
                horizon=HORIZON
            )
            
            self.model.fit(
                X, y,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                verbose=0
            )
            
        except Exception as e:
            logger.error("Error training LSTM model", exc_info=True)
            raise

    def calculate_signals(self, company_dict: Dict) -> pd.DataFrame:
        """Generate trading signals based on predicted price direction"""
        logger.info("Calculating LSTM direction-based signals")
        
        try:
            # Train on training data
            self.train(company_dict['train'], company_dict['scaler'])
            
            # Generate signals for test data
            df = company_dict['test'].copy()
            df['Signal'] = 'HOLD'
            position_open = False
            
            for i in range(len(df) - WINDOW_SIZE - HORIZON + 1):
                window = df['Price'].iloc[i:i+WINDOW_SIZE].values
                X = company_dict['scaler'].transform(window.reshape(-1, 1)).reshape(1, WINDOW_SIZE, 1)
                
                pred_scaled = self.model.predict(X, verbose=0)[0]
                predictions = company_dict['scaler'].inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                
                current_price = df['Price'].iloc[i + WINDOW_SIZE - 1]
                predicted_return = (predictions[-1] - current_price) / current_price
                
                idx = df.index[i + WINDOW_SIZE - 1]
                
                if not position_open:
                    if predicted_return > MIN_RETURN:
                        df.loc[idx, 'Signal'] = 'BUY'
                        position_open = True
                else:
                    if predicted_return < -MIN_RETURN:
                        df.loc[idx, 'Signal'] = 'SELL'
                        position_open = False
            
            logger.info(f"Generated signals: {df['Signal'].value_counts().to_dict()}")
            return df
            
        except Exception as e:
            logger.error("Error calculating LSTM signals", exc_info=True)
            raise

class NaiveForecastStrategy(BaseStrategy):
    """Naive forecast strategy that assumes tomorrow's price will be the same as today"""
    
    def __init__(self):
        super().__init__("Naive Forecast")
    
    def calculate_signals(self, company_dict: Dict) -> pd.DataFrame:
        """Generate trading signals based on naive forecast"""
        logger.info("Calculating naive forecast signals")
        
        try:
            # Only use test data
            df = company_dict['test'].copy()
            df['Signal'] = 'HOLD'
            position_open = False
            
            # Calculate daily returns (this tells us if we were wrong about prices staying the same)
            df['Daily_Return'] = df['Price'].pct_change()
            
            for i in range(len(df) - 1):  # -1 because we can't trade on last day
                if not position_open:
                    # If price went up when we predicted flat, maybe trend is starting
                    if df['Daily_Return'].iloc[i] > MIN_RETURN:
                        df.iloc[i, df.columns.get_loc('Signal')] = 'BUY'
                        position_open = True
                else:
                    # If price went down when we predicted flat, maybe trend is ending
                    if df['Daily_Return'].iloc[i] < -MIN_RETURN:
                        df.iloc[i, df.columns.get_loc('Signal')] = 'SELL'
                        position_open = False
            
            logger.info(f"Generated signals: {df['Signal'].value_counts().to_dict()}")
            return df
            
        except Exception as e:
            logger.error("Error calculating naive forecast signals", exc_info=True)
            raise

def compare_strategies(company_dict: Dict) -> Dict:
    """Compare all strategies on test data"""
    logger.info("Starting strategy comparison")
    
    try:
        strategies = {
            'bollinger': BollingerBandStrategy(),
            'lstm': LSTMStrategy(),
            'naive': NaiveForecastStrategy()
        }
        
        results = {}
        for name, strategy in strategies.items():
            results[name] = strategy.analyze(company_dict)
            
        # Find best performing strategy
        best_strategy = max(
            results.items(),
            key=lambda x: x[1]['returns']['total_return']
        )[0]
        
        # Find best risk-adjusted strategy
        best_risk_adjusted = max(
            results.items(),
            key=lambda x: x[1]['metrics']['sharpe_ratio']
        )[0]
        
        comparison = {
            'results': results,
            'best_strategy': best_strategy,
            'best_risk_adjusted': best_risk_adjusted
        }
        
        logger.info(f"Best performing strategy: {best_strategy}")
        return comparison
        
    except Exception as e:
        logger.error("Error comparing strategies", exc_info=True)
        raise