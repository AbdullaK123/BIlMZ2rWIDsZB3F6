import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
from value_investor.logging import logger
from value_investor.config import DATA_DIR


def fetch_exchange_rates(start_date: str, end_date: str, output_file: str):
    """
    Fetch historical exchange rates from Yahoo Finance and save to CSV
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_file: Name of output CSV file
    """
    # Currency pairs in Yahoo Finance format (USD as base currency)
    currency_pairs = {
        'RUB': 'USDRUB=X',    # USD/Russian Ruble
        'TRY': 'USDTRY=X',    # USD/Turkish Lira
        'EGP': 'USDEGP=X',    # USD/Egyptian Pound
        'BRL': 'USDBRL=X',    # USD/Brazilian Real
        'ARS': 'USDARS=X',    # USD/Argentine Peso
        'COP': 'USDCOP=X',    # USD/Colombian Peso
        'ZAR': 'USDZAR=X',    # USD/South African Rand
        'KRW': 'USDKRW=X'     # USD/Korean Won
    }
    
    logger.info(f"Fetching exchange rates from {start_date} to {end_date}")
    
    all_data = []
    
    # Get historical data for all currencies
    for currency, symbol in currency_pairs.items():
        try:
            logger.info(f"Fetching {currency} rates...")
            
            # Get historical data
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if df.empty:
                logger.error(f"No data received for {currency}")
                continue
                
            # Keep only the Close price and rename column
            df = df[['Close']].rename(columns={'Close': currency})
            all_data.append(df)
            
            logger.info(f"Successfully fetched {len(df)} days of {currency} rates")
            
            # Add small delay to avoid hitting rate limits
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error fetching {currency}: {str(e)}")
    
    # Combine all currency data
    if all_data:
        rates_df = pd.concat(all_data, axis=1)
        rates_df = rates_df.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate statistics
        stats = rates_df.agg(['mean', 'std', 'min', 'max'])
        volatility = rates_df.pct_change().std() * 100
        max_drawdown = ((rates_df - rates_df.expanding().max()) / 
                       rates_df.expanding().max()).min() * 100
        
        # Log statistics
        logger.info("\nExchange Rate Statistics:")
        logger.info("\nMean Rates:")
        for curr in rates_df.columns:
            logger.info(f"{curr}: {stats.loc['mean', curr]:.4f}")
        
        logger.info("\nVolatility (%):")
        for curr in rates_df.columns:
            logger.info(f"{curr}: {volatility[curr]:.2f}%")
        
        logger.info("\nMax Drawdown (%):")
        for curr in rates_df.columns:
            logger.info(f"{curr}: {max_drawdown[curr]:.2f}%")
        
        # Save to CSV
        output_path = DATA_DIR / output_file
        rates_df.to_csv(output_path)
        logger.info(f"\nSaved exchange rates to {output_path}")
        
        return rates_df
    else:
        logger.error("No exchange rate data was retrieved")
        return None

if __name__ == "__main__":
    try:
        # Fetch rates for 2020 to Q1 2021
        rates_df = fetch_exchange_rates(
            start_date='2020-01-01',
            end_date='2021-04-01',
            output_file='exchange_rates_2020_2021Q1.csv'
        )
        
        if rates_df is None:
            logger.error("Failed to fetch exchange rates")
            
    except Exception as e:
        logger.error(f"Script failed: {str(e)}", exc_info=True)