import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List
from value_investor.preprocessing import get_stock_data
from value_investor.strategies import (
    BollingerBandStrategy, 
    LSTMStrategy,
    NaiveForecastStrategy,
    compare_strategies
)
from value_investor.config import (
    DATA_FILE,
    INITIAL_CAPITAL,
    OUTPUT_DIR
)
from value_investor.logging import logger, metrics_logger

def setup_args() -> argparse.Namespace:
    """Set up command line arguments"""
    parser = argparse.ArgumentParser(description='Value Investor Trading System')
    parser.add_argument(
        '--strategy',
        choices=['bollinger', 'lstm', 'naive', 'all'],
        default='all',
        help='Trading strategy to use'
    )
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=INITIAL_CAPITAL,
        help='Initial investment amount'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='results.json',
        help='Output file for results'
    )
    
    return parser.parse_args()

def get_strategy(strategy_name: str):
    """Get strategy instance based on name"""
    strategy_map = {
        'bollinger': BollingerBandStrategy(),
        'lstm': LSTMStrategy(),
        'naive': NaiveForecastStrategy()
    }
    return strategy_map.get(strategy_name)

def calculate_portfolio_metrics(results: Dict) -> Dict:
    """Calculate aggregate portfolio metrics"""
    try:
        returns = []
        trades = []
        sharpe_ratios = []
        max_drawdowns = []
        win_rates = []
        
        for company_results in results.values():
            returns.append(company_results['returns']['total_return'])
            trades.append(company_results['returns']['number_of_trades'])
            sharpe_ratios.append(company_results['metrics']['sharpe_ratio'])
            max_drawdowns.append(company_results['returns']['max_drawdown'])
            win_rates.append(company_results['returns']['win_rate'])
        
        portfolio_metrics = {
            'avg_return': sum(returns) / len(returns),
            'best_return': max(returns),
            'worst_return': min(returns),
            'avg_trades': sum(trades) / len(trades),
            'avg_sharpe_ratio': sum(sharpe_ratios) / len(sharpe_ratios),
            'avg_max_drawdown': sum(max_drawdowns) / len(max_drawdowns),
            'avg_win_rate': sum(win_rates) / len(win_rates),
            'companies_analyzed': len(results)
        }
        
        return portfolio_metrics
        
    except Exception as e:
        logger.error("Error calculating portfolio metrics", exc_info=True)
        raise

def analyze_single_strategy(
    strategy_name: str,
    stock_data: Dict,
    initial_capital: float
) -> Dict:
    """
    Analyze a single strategy across all stocks.
    
    Args:
        strategy_name: Name of strategy to use
        stock_data: Dictionary of stock data
        initial_capital: Initial investment amount
        
    Returns:
        Dictionary containing analysis results
    """
    logger.info(f"Analyzing {strategy_name} strategy")
    
    try:
        strategy = get_strategy(strategy_name)
        if not strategy:
            raise ValueError(f"Invalid strategy name: {strategy_name}")
            
        results = {}
        for company, data in stock_data.items():
            logger.info(f"Analyzing {company} with {strategy_name} strategy")
            
            try:
                # Run analysis on test data
                results[company] = strategy.analyze(data, initial_capital)
                
                # Log results
                metrics_logger.save_metrics(
                    metrics={
                        'return': results[company]['returns']['total_return'],
                        'trades': results[company]['returns']['number_of_trades'],
                        'sharpe_ratio': results[company]['metrics']['sharpe_ratio'],
                        'max_drawdown': results[company]['returns']['max_drawdown'],
                        'win_rate': results[company]['returns']['win_rate']
                    },
                    category='company_performance',
                    strategy=f"{strategy_name}_{company}"
                )
                
                logger.info(
                    f"{company} Results - "
                    f"Return: {results[company]['returns']['total_return']:.2f}%, "
                    f"Sharpe: {results[company]['metrics']['sharpe_ratio']:.2f}"
                )
                
            except Exception as e:
                logger.error(f"Error analyzing {company}", exc_info=True)
                continue
                
        return results
        
    except Exception as e:
        logger.error(f"Error in strategy analysis: {str(e)}", exc_info=True)
        raise

def main():
    """Main execution function"""
    try:
        # Parse command line arguments
        args = setup_args()
        logger.info(f"Starting analysis with strategy={args.strategy}")
        
        # Load stock data
        logger.info(f"Loading stock data from {DATA_FILE}")
        stock_data = get_stock_data(DATA_FILE)
        
        if not stock_data:
            logger.error("No stock data loaded")
            return
            
        logger.info(f"Successfully loaded data for {len(stock_data)} companies")
        
        # Initialize results
        all_results = {}
        
        # Run analysis
        if args.strategy == 'all':
            # Run all strategies
            for strategy_name in ['bollinger', 'lstm', 'naive']:
                logger.info(f"Running {strategy_name} strategy")
                all_results[strategy_name] = analyze_single_strategy(
                    strategy_name,
                    stock_data,
                    args.initial_capital
                )
                
            # Compare strategies
            comparison_results = compare_strategies(stock_data)
            all_results['comparison'] = comparison_results
            
        else:
            # Run single strategy
            all_results[args.strategy] = analyze_single_strategy(
                args.strategy,
                stock_data,
                args.initial_capital
            )
        
        # Calculate and log portfolio metrics
        for strategy_name, strategy_results in all_results.items():
            if strategy_name != 'comparison':
                portfolio_metrics = calculate_portfolio_metrics(strategy_results)
                
                metrics_logger.save_metrics(
                    metrics=portfolio_metrics,
                    category='portfolio_performance',
                    strategy=strategy_name
                )
                
                logger.info(
                    f"\n{strategy_name} Portfolio Performance:"
                    f"\nAverage Return: {portfolio_metrics['avg_return']:.2f}%"
                    f"\nBest Return: {portfolio_metrics['best_return']:.2f}%"
                    f"\nWorst Return: {portfolio_metrics['worst_return']:.2f}%"
                    f"\nAverage Sharpe: {portfolio_metrics['avg_sharpe_ratio']:.2f}"
                    f"\nAverage Win Rate: {portfolio_metrics['avg_win_rate']:.1f}%"
                    f"\nAverage Max Drawdown: {portfolio_metrics['avg_max_drawdown']:.1f}%"
                )
        
        # Save results
        output_path = OUTPUT_DIR / args.output_file
        metrics_logger.save_results(
            results=all_results,
            experiment_name='trading_analysis'
        )
        
        logger.info(f"Results saved to {output_path}")
        
    except Exception as e:
        logger.error("Error in main execution", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        main()
        logger.info("Program completed successfully")
    except Exception as e:
        logger.error("Program failed", exc_info=True)