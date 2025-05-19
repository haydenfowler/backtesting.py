import yfinance as yf
import pandas as pd

def load_yahoo_finance_data(symbol, start_date=None, end_date=None, period=None, interval='1d'):
    """
    Download and prepare price data from Yahoo Finance.
    
    Args:
        symbol (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        start_date (str, optional): Start date in format 'YYYY-MM-DD'
        end_date (str, optional): End date in format 'YYYY-MM-DD'
        period (str, optional): Alternative to start/end dates. Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        interval (str, optional): Data interval. Default '1d' (daily). Options: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data ready for backtesting
        
    Note:
        Either provide both start_date and end_date OR provide period, but not both.
    """
    # Validate input parameters
    if period and (start_date or end_date):
        raise ValueError("Cannot specify both period and start/end dates. Use either period OR date range.")
    
    if not period and not (start_date and end_date):
        raise ValueError("Must provide either period OR both start_date and end_date.")
        
    # Download data from Yahoo Finance using either method
    if period:
        data = yf.download(symbol, period=period, interval=interval)
    else:
        data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    
    # Fix yfinance data format for backtesting compatibility
    if isinstance(data.columns, pd.MultiIndex):
        # Get the first level column names (Open, High, Low, etc.)
        data.columns = data.columns.get_level_values(0)
    
    # Ensure we have the minimum required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_columns[:4]):  # Open, High, Low, Close are required
        raise ValueError(f"Data missing required columns. Available columns: {data.columns.tolist()}")
    
    return data

def load_multiple_symbols(symbols, start_date=None, end_date=None, period=None, interval='1d'):
    """
    Download data for multiple symbols and return as a dictionary of DataFrames.
    
    Args:
        symbols (list): List of stock ticker symbols
        start_date (str, optional): Start date in format 'YYYY-MM-DD'
        end_date (str, optional): End date in format 'YYYY-MM-DD'
        period (str, optional): Alternative to start/end dates. Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        interval (str, optional): Data interval. Default '1d' (daily)
        
    Returns:
        dict: Dictionary mapping symbols to their respective DataFrames
    """
    data_dict = {}
    for symbol in symbols:
        data_dict[symbol] = load_yahoo_finance_data(
            symbol, 
            start_date=start_date, 
            end_date=end_date, 
            period=period, 
            interval=interval
        )
    return data_dict 