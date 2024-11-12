import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.preprocessing import MinMaxScaler
from value_investor.config import (
    DATA_DIR, WINDOW_SIZE, HORIZON, 
    EXCHANGE_RATES_FILE, TRAIN_TEST_SPLIT_DATE
)
from value_investor.logging import logger, metrics_logger

def convert_to_float(x: str) -> float:
    """
    Convert string values with K, M suffixes to float.
    Example: '1.5M' -> 1500000, '1.5K' -> 1500
    """
    try:
        if isinstance(x, (int, float)):
            return float(x)
        if 'M' in x:
            return float(x.replace('M', '')) * 1000000
        elif 'K' in x:
            return float(x.replace('K', '')) * 1000
        elif '-' in x:  # Handle missing values
            return 0
        else:
            return float(x)
    except (ValueError, TypeError) as e:
        logger.error(f"Failed to convert value '{x}' to float", exc_info=True)
        raise

def load_exchange_rates() -> pd.DataFrame:
    """
    Load and process exchange rates from CSV file.
    Returns DataFrame with currency exchange rates indexed by date.
    """
    try:
        logger.info(f"Loading exchange rates from {EXCHANGE_RATES_FILE}")
        
        # Read CSV with explicitly defined date column
        rates_df = pd.read_csv(
            EXCHANGE_RATES_FILE, 
            skiprows=3,  # Skip Price,Ticker,Date headers
            names=['Date', 'RUB', 'TRY', 'EGP', 'BRL', 'ARS', 'COP', 'ZAR', 'KRW']
        )
        
        # Convert first column to datetime and remove timezone info
        rates_df['Date'] = pd.to_datetime(rates_df['Date']).dt.tz_localize(None)
        
        # Set Date as index
        rates_df = rates_df.set_index('Date')
        
        # Sort by date
        rates_df = rates_df.sort_index()
        
        # Convert all columns to numeric
        for col in rates_df.columns:
            rates_df[col] = pd.to_numeric(rates_df[col], errors='coerce')
        
        # Forward fill any missing values
        rates_df = rates_df.ffill()
        
        if rates_df.empty:
            raise ValueError("Exchange rates dataframe is empty after processing")
            
        # Verify we have all required currencies
        required_currencies = {'RUB', 'TRY', 'EGP', 'BRL', 'ARS', 'COP', 'ZAR', 'KRW'}
        missing_currencies = required_currencies - set(rates_df.columns)
        if missing_currencies:
            raise ValueError(f"Missing required currencies: {missing_currencies}")
        
        logger.info(
            f"Successfully loaded rates from {rates_df.index.min()} "
            f"to {rates_df.index.max()}"
        )
        
        return rates_df
        
    except Exception as e:
        logger.error(f"Error loading exchange rates: {str(e)}", exc_info=True)
        raise

def process_stock_data(df: pd.DataFrame, company: str) -> pd.DataFrame:
    """
    Process raw stock data for a single company.
    Handles data cleaning, conversion, and validation.
    """
    try:
        df = df.copy()
        
        # Drop last incomplete row if exists
        df = df.iloc[:-1] if not df.empty else df
        
        # Convert to numeric values
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['Vol.'] = df['Vol.'].apply(convert_to_float)
        
        # Log data quality issues
        nan_prices = df['Price'].isna().sum()
        zero_volume = (df['Vol.'] == 0).sum()
        if nan_prices > 0:
            logger.warning(f"{company}: Found {nan_prices} NaN prices")
        if zero_volume > 0:
            logger.warning(f"{company}: Found {zero_volume} zero volume days")
        
        # Set up datetime index - ensure timezone naive
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = df.set_index('Date').sort_index()
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing {company}: {str(e)}", exc_info=True)
        raise

def convert_to_usd(df: pd.DataFrame, currency: str, exchange_rates: pd.DataFrame) -> pd.DataFrame:
    """
    Convert price columns from local currency to USD using exchange rates.
    """
    try:
        df = df.copy()
        price_columns = ['Price', 'Open', 'High', 'Low', 'Close']
        
        # Ensure indices are aligned and timezone naive
        common_dates = df.index.intersection(exchange_rates.index)
        df = df.loc[common_dates]
        rates = exchange_rates.loc[common_dates]
        
        # Convert each price column
        for col in price_columns:
            if col in df.columns:
                orig_prices = df[col].copy()
                df[col] = df[col] / rates[currency]
                
                # Log conversion statistics
                logger.debug(
                    f"{currency} {col} conversion - "
                    f"Mean before: {orig_prices.mean():.2f}, "
                    f"Mean after: {df[col].mean():.2f} USD"
                )
        
        return df
        
    except Exception as e:
        logger.error(f"Error converting {currency} to USD: {str(e)}", exc_info=True)
        raise

def create_sequences(
    data: np.ndarray,
    window_size: int = WINDOW_SIZE,
    horizon: int = HORIZON
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction.
    Returns features (X) and targets (y) arrays.
    """
    try:
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        data = data.reshape(-1, 1)
        X, y = [], []
        
        for i in range(len(data) - window_size - horizon + 1):
            X.append(data[i:(i + window_size)])
            y.append(data[(i + window_size):(i + window_size + horizon)])
        
        X = np.array(X)
        y = np.array(y)
        
        # Ensure correct shapes
        X = X.reshape(X.shape[0], window_size, 1)
        y = y.reshape(y.shape[0], horizon)
        
        logger.debug(f"Created sequences - X: {X.shape}, y: {y.shape}")
        return X, y
        
    except Exception as e:
        logger.error("Error creating sequences", exc_info=True)
        raise

def get_stock_data(file_path: str) -> Dict:
    """
    Load and process all stock data from Excel file.
    Returns dictionary of processed DataFrames for each company.
    """
    try:
        # Load exchange rates first
        exchange_rates = load_exchange_rates()
        
        # Read Excel file
        excel = pd.ExcelFile(file_path)
        company_names = excel.sheet_names
        
        logger.info(f"Processing {len(company_names)} companies")
        
        # Currency mapping
        currency_map = {
            'Russia': 'RUB',
            'Turkey': 'TRY',
            'Egypt': 'EGP',
            'Brazil': 'BRL',
            'Argentina': 'ARS',
            'Colombia': 'COP',
            'South Africa': 'ZAR',
            'South Korea': 'KRW'
        }
        
        dataframes = {}
        for company in company_names:
            try:
                # Load and process data
                df = pd.read_excel(excel, company)
                df = process_stock_data(df, company)
                
                # Convert to USD
                country = company.split('-')[0].strip()
                currency = currency_map.get(country)
                
                if not currency:
                    logger.error(f"No currency mapping for {country}")
                    continue
                    
                logger.info(f"Converting {currency} to USD for {company}")
                df = convert_to_usd(df, currency, exchange_rates)
                
                # Split data
                split_date = pd.to_datetime(TRAIN_TEST_SPLIT_DATE)
                train_data = df[df.index < split_date]
                test_data = df[df.index >= split_date]
                
                # Normalize price data
                scaler = MinMaxScaler()
                train_prices = scaler.fit_transform(train_data['Price'].values.reshape(-1, 1))
                test_prices = scaler.transform(test_data['Price'].values.reshape(-1, 1))
                
                dataframes[company] = {
                    'train': train_data,
                    'test': test_data,
                    'train_prices': train_prices,
                    'test_prices': test_prices,
                    'scaler': scaler
                }
                
                logger.info(
                    f"{company}: Split into {len(train_data)} train "
                    f"and {len(test_data)} test samples"
                )
                
            except Exception as e:
                logger.error(f"Error processing {company}", exc_info=True)
                continue
        
        return dataframes
        
    except Exception as e:
        logger.error(f"Error loading stock data: {str(e)}", exc_info=True)
        raise