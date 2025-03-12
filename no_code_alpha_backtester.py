#!/usr/bin/env python3  # Tells the operating system to run this script with the python3 interpreter

"""
Robust No-Code SP500 Portfolio Backtester with Performance Metrics and Result Saving

This engine:
  • Downloads daily OHLCV data for all S&P 500 stocks using yfinance.
  • Computes a custom alpha for each stock (using a wide range of primitives, e.g., abs, delta, SMA, ts_rank, etc.)
    based on a user-specified formula.
  • Applies a cross-sectional filter so that only stocks in the top percentile (e.g., top 10% when cs_percentile=0.9)
    on each day generate a trade signal.
  • Simulates trades: if a stock signals on day i, trade on day i+1 (buy at open, sell at close).
  • Aggregates daily trade returns into a portfolio equity curve.
  • Computes performance metrics (annualized return, volatility, Sharpe ratio, max drawdown, hypothesis tests, confidence intervals,
    and a Monte Carlo permutation test).
  • Saves a summary of all results to a text file ("backtest_results.txt").

DISCLAIMER: Use at your own risk. Trading is inherently risky.
"""
# This multi-line string describes the purpose of the script and serves as documentation.

import re               # Import the regular expressions module for text processing
import numpy as np      # Import NumPy for numerical operations (aliased as np)
import pandas as pd     # Import pandas for data manipulation and analysis (aliased as pd)
import matplotlib.pyplot as plt   # Import matplotlib's plotting library (aliased as plt)
from scipy.stats import rankdata, skew, kurtosis, kendalltau, ttest_1samp, sem, wilcoxon, binomtest, t  
# Import statistical functions and tests from scipy.stats used for performance metrics
import yfinance as yf   # Import yfinance for downloading stock market data (aliased as yf)

import os
import pickle
from datetime import datetime

def get_yfinance_data(tickers, start_date, end_date, cache_filename="sp500_yf_data.pkl", force_update=False):
    # Convert date strings to datetime objects for easy comparison
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Check for an existing cache file unless a force update is requested
    if os.path.exists(cache_filename) and not force_update:
        with open(cache_filename, "rb") as f:
            cached_data, cached_start, cached_end = pickle.load(f)
        # Parse the cached date range
        cached_start_dt = datetime.strptime(cached_start, "%Y-%m-%d")
        cached_end_dt = datetime.strptime(cached_end, "%Y-%m-%d")
        if cached_start_dt <= start_dt and cached_end_dt >= end_dt:
            print("Using cached yfinance data from {} to {}.".format(cached_start, cached_end))
            return cached_data
        else:
            print("Cached data does not cover the requested range. Updating cache...")
    else:
        print("No cached data found. Downloading from yfinance...")

    # Download data using yfinance
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True, progress=False)
    # Save the data along with the date range for future use
    with open(cache_filename, "wb") as f:
        pickle.dump((data, start_date, end_date), f)
    return data

#############################################
# PERFORMANCE METRICS: AlphaTester Class
#############################################
class AlphaTester:
    def __init__(self, returns, trading_days=252, risk_free_rate=0.0):
        self.returns = np.array(returns)   # Convert input returns to a NumPy array for numerical calculations
        self.trading_days = trading_days   # Set the number of trading days in a year (default 252)
        self.risk_free_rate = risk_free_rate  # Set the risk-free rate used in Sharpe ratio calculation

    def annualized_return(self):
        cum_return = np.prod(1 + self.returns) - 1  # Calculate the cumulative return by compounding daily returns
        n = len(self.returns)   # Number of return observations (trading days)
        return (1 + cum_return)**(self.trading_days / n) - 1  # Convert cumulative return into an annualized return

    def annualized_volatility(self):
        return np.std(self.returns, ddof=1) * np.sqrt(self.trading_days)  # Annualize the standard deviation of daily returns

    def sharpe_ratio(self):
        ann_ret = self.annualized_return()  # Compute the annualized return
        ann_vol = self.annualized_volatility()  # Compute the annualized volatility
        # Calculate the Sharpe ratio (excess return per unit volatility) and handle division by zero
        return (ann_ret - self.risk_free_rate) / ann_vol if ann_vol != 0 else np.nan

    def max_drawdown(self):
        cumulative = np.cumprod(1 + self.returns)  # Compute the running product of returns to get cumulative equity
        running_max = np.maximum.accumulate(cumulative)  # Determine the maximum value reached up to each point
        drawdowns = (running_max - cumulative) / running_max  # Calculate the percentage drop from the running maximum
        return np.max(drawdowns)  # Return the worst (maximum) drawdown observed

    def t_test(self):
        t_stat, p_value = ttest_1samp(self.returns, popmean=0)  # Perform a one-sample t-test (mean = 0)
        return t_stat, p_value  # Return the test statistic and its p-value

    def wilcoxon_test(self):
        nonzero = self.returns[self.returns != 0]  # Filter out zero returns to avoid skewing the test
        if len(nonzero) < 10:
            return np.nan  # Not enough data points to perform the test reliably
        stat, p_value = wilcoxon(nonzero)  # Execute the Wilcoxon signed-rank test on the nonzero returns
        return p_value  # Return only the p-value from the test

    def sign_test(self):
        positives = np.sum(self.returns > 0)  # Count how many returns are positive
        negatives = np.sum(self.returns <= 0)  # Count how many returns are zero or negative
        n = positives + negatives  # Total number of valid returns considered
        # Perform a binomial test to see if the proportion of positive returns significantly deviates from 0.5
        return binomtest(positives, n, p=0.5, alternative='two-sided').pvalue

    def confidence_interval(self, confidence=0.95):
        n = len(self.returns)  # Number of return observations
        mean_return = np.mean(self.returns)  # Calculate the average return
        std_err = sem(self.returns)  # Compute the standard error of the mean
        t_crit = t.ppf((1 + confidence) / 2., n-1)  # Find the t-critical value for the desired confidence level
        margin = t_crit * std_err  # Determine the margin of error
        return (mean_return - margin, mean_return + margin)  # Return the lower and upper bounds of the confidence interval

    def permutation_test(self, n_permutations=1000, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)  # Set a random seed for reproducibility if provided
        observed_mean = np.mean(self.returns)  # Calculate the actual mean return
        # Generate a distribution of means by randomly permuting the returns repeatedly
        perm_means = [np.mean(np.random.permutation(self.returns)) for _ in range(n_permutations)]
        perm_means = np.array(perm_means)  # Convert the list of means to a NumPy array for easier computation
        # Compute the p-value as the fraction of permutations where the absolute mean is at least as extreme as observed
        return np.mean(np.abs(perm_means) >= np.abs(observed_mean))

    def summary(self):
        # Aggregate all the performance metrics into a single dictionary for easy access
        return {
            "Annualized Return": self.annualized_return(),
            "Annualized Volatility": self.annualized_volatility(),
            "Sharpe Ratio": self.sharpe_ratio(),
            "Max Drawdown": self.max_drawdown(),
            "T-test p-value": self.t_test()[1],
            "Wilcoxon p-value": self.wilcoxon_test(),
            "Sign Test p-value": self.sign_test(),
            "95% CI for Mean Return": self.confidence_interval(),
            "Permutation Test p-value": self.permutation_test()
        }

#############################################
# --- Primitives and Helper Functions ---
#############################################

# Simple Element-wise Operations
def neg(x):
    return -1 * x  # Return the negative of x (simple negation)

def recpcal(x):
    return 1 / x  # Calculate the reciprocal (1 divided by x)

def powa(x, a):
    return np.power(x, a)  # Raise x to the power a using NumPy's power function

# Cross-sectional, Rank, Standardize Operations
def csrank(x):
    return rankdata(x, method="average", nan_policy="omit")  # Rank elements of x, averaging ranks in case of ties

def cszscre(x):
    return (x - np.mean(x)) / np.std(x)  # Standardize x to have mean 0 and std 1

# Time-series, Difference, Moving Avergae, Rank with Window Operations 
def delta(series, period):
    return series.diff(period)  # Compute the difference between current and lagged values by the specified period

def delta_a(x, a):
    try:
        return x.iloc[-1] - x.iloc[-(a+1)]  # Calculate the difference between the last value and the value a steps back
    except Exception:
        return np.nan  # Return NaN if the operation fails (e.g., insufficient data)

def delay(series, period):
    return series.shift(period)  # Shift the series by the specified number of periods (delaying the data)

def delay_a(x, a):
    try:
        return x.iloc[-(a+1)]  # Retrieve the value a steps before the last element
    except Exception:
        return np.nan  # Return NaN if index is out of bounds

def SMA(series, window):
    return series.rolling(window, min_periods=window).mean()  # Compute the Simple Moving Average over the given window

def ts_rank(series, window):
    # Apply a rolling window to compute the rank of the latest value in each window
    return series.rolling(window, min_periods=window).apply(lambda x: rankdata(x, method="average")[-1], raw=False)

def tsrank_a(x):
    try:
        return rankdata(x, method="average", nan_policy="omit")[-1]  # Return the rank of the last element in x
    except Exception:
        return np.nan

def tsmax_a(x):
    return np.max(x)  # Return the maximum value in x

def tsmin_a(x):
    return np.min(x)  # Return the minimum value in x

def tsargmax_a(x):
    return np.argmax(x)  # Return the index of the maximum value in x

def tsargmin_a(x):
    return np.argmin(x)  # Return the index of the minimum value in x

def tszscre_a(x):
    try:
        return (x.iloc[-1] - np.mean(x)) / np.std(x)  # Standardize the last element of x based on the rest of the series
    except Exception:
        return np.nan

# Aggregation, multi data points into a single value, Operations
def sum_a(x):
    return np.sum(x)  # Compute the sum of the elements in x

def prod_a(x):
    return np.prod(x)  # Compute the product of the elements in x

def mean_a(x):
    return np.mean(x)  # Compute the mean of x

def ewma_a(x, alpha=0.36):
    return x.ewm(alpha=alpha, adjust=False).mean()  # Compute the Exponentially Weighted Moving Average of x

def median_a(x):
    return np.median(x)  # Compute the median of x

def std_a(x):
    return np.std(x)  # Compute the standard deviation of x

def var_a(x):
    return np.var(x)  # Compute the variance of x

def skew_a(x):
    try:
        return skew(x)  # Compute the skewness of x
    except Exception:
        return np.nan

def kurt_a(x):
    try:
        return kurtosis(x, fisher=True)  # Compute the kurtosis (Fisher's definition) of x
    except Exception:
        return np.nan

# Vectorized Operations
def max_func(*args):
    arrays = [np.array(arg) for arg in args]  # Convert all inputs into NumPy arrays
    return np.maximum.reduce(arrays)  # Compute the element-wise maximum across all arrays

def plus(x, y):
    return x + y  # Add x and y element-wise

def minus(x, y):
    return x - y  # Subtract y from x element-wise

def mult(x, y):
    return x * y  # Multiply x and y element-wise

def div(x, y):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = x / y  # Divide x by y element-wise
        result[~np.isfinite(result)] = np.nan  # Replace infinities or invalid numbers with NaN
    return result

def and_func(x, y):
    return np.logical_and(x, y)  # Perform element-wise logical AND on x and y

def or_func(x, y):
    return np.logical_or(x, y)  # Perform element-wise logical OR on x and y

def eq(x, y):
    return x == y  # Return element-wise equality comparison between x and y

def gt(x, y):
    return x > y  # Return element-wise greater-than comparison

def lt(x, y):
    return x < y  # Return element-wise less-than comparison

def ite(x, y, z):
    return np.where(x, y, z)  # If condition x is true, choose y; otherwise, choose z (element-wise)

# Pairwise Opertaions
def cor_a(x, y):
    try:
        return np.corrcoef(x, y)[0,1]  # Compute Pearson correlation between x and y
    except Exception:
        return np.nan

def kentau_a(x, y):
    try:
        return kendalltau(x, y)[0]  # Compute Kendall's tau correlation between x and y
    except Exception:
        return np.nan

def cov_a(x, y):
    try:
        return np.cov(x, y)[0,1]  # Compute the covariance between x and y
    except Exception:
        return np.nan

#############################################
# Preprocessing and Formula Evaluation
#############################################
def preprocess_formula(formula):
    variables = ['open', 'high', 'low', 'close', 'volume']  # Define the variable names to standardize
    for var in variables:
        # Replace each occurrence of a variable (case-insensitive) with its uppercase version
        formula = re.sub(r'\b' + var + r'\b', var.upper(), formula, flags=re.IGNORECASE)
    return formula  # Return the modified formula

def compute_user_alpha(df, formula):
    formula = preprocess_formula(formula)  # Preprocess the formula to ensure consistent variable names
    safe_env = {
        'OPEN': df['Open'],    # Map the DataFrame's 'Open' column to the variable OPEN
        'HIGH': df['High'],    # Map the DataFrame's 'High' column to HIGH
        'LOW': df['Low'],      # Map 'Low' to LOW
        'CLOSE': df['Close'],  # Map 'Close' to CLOSE
        'VOLUME': df['Volume'],# Map 'Volume' to VOLUME
        'abs': np.abs,         # Include the absolute value function
        'neg': neg,            # Include our negation function
        'log': np.log,         # Include the natural logarithm function
        'sign': np.sign,       # Include the sign function
        'recpcal': recpcal,    # Include the reciprocal function
        'pow': powa,           # Include the power function
        'csrank': csrank,      # Include cross-sectional ranking
        'cszscre': cszscre,    # Include cross-sectional z-score standardization
        'delta': delta,        # Include the delta (difference) function
        'delta_a': delta_a,    # Include the alternative delta function
        'delay': delay,        # Include the delay (shift) function
        'delay_a': delay_a,    # Include the alternative delay function
        'SMA': SMA,            # Include the Simple Moving Average function
        'ts_rank': ts_rank,    # Include the time-series rank function
        'tsrank_a': tsrank_a,  # Include the alternative time-series rank function
        'tsmax_a': tsmax_a,    # Include the time-series maximum function
        'tsmin_a': tsmin_a,    # Include the time-series minimum function
        'tsargmax_a': tsargmax_a,  # Include the function to find index of maximum value
        'tsargmin_a': tsargmin_a,  # Include the function to find index of minimum value
        'tszscre_a': tszscre_a,    # Include the function for time-series z-score of the last value
        'sum_a': sum_a,        # Include the summation function
        'prod_a': prod_a,      # Include the product function
        'mean_a': mean_a,      # Include the mean calculation function
        'ewma_a': ewma_a,      # Include the EWMA function
        'median_a': median_a,  # Include the median calculation function
        'std_a': std_a,        # Include the standard deviation function
        'var_a': var_a,        # Include the variance function
        'skew_a': skew_a,      # Include the skewness function
        'kurt_a': kurt_a,      # Include the kurtosis function
        'max': max_func,       # Include the maximum function for multiple arrays
        'plus': plus,          # Include the addition operator function
        'minus': minus,        # Include the subtraction operator function
        'mult': mult,          # Include the multiplication operator function
        'div': div,            # Include the division operator function
        'and': and_func,       # Include the logical AND function
        'or': or_func,         # Include the logical OR function
        'eq': eq,              # Include the equality operator function
        'gt': gt,              # Include the greater-than operator function
        'lt': lt,              # Include the less-than operator function
        'ite': ite,            # Include the if-then-else function
        'cor_a': cor_a,        # Include the correlation function
        'kentau_a': kentau_a,  # Include the Kendall tau function
        'cov_a': cov_a,        # Include the covariance function
        'np': np,              # Provide access to NumPy functions if needed in the formula
    }
    try:
        # Evaluate the formula in a restricted environment (to avoid security issues)
        result = eval(formula, {"__builtins__": {}}, safe_env)
    except Exception as e:
        print("Error evaluating formula:", e)  # Print an error if evaluation fails
        result = pd.Series(np.nan, index=df.index)  # Return a Series of NaN values if something goes wrong
    return result  # Return the computed alpha values for the DataFrame

#############################################
# Portfolio Backtesting for SP500 Stocks
#############################################
def portfolio_backtest_sp500(alpha_formula, cs_percentile, start_date, end_date):
    try:
        # Read the list of S&P 500 companies from Wikipedia
        sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        tickers = sp500_table[0]['Symbol'].tolist()  # Extract ticker symbols from the first table
    except Exception as e:
        print("Error obtaining S&P 500 tickers:", e)
        return None, None  # Return if the tickers cannot be retrieved

    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")
    # Download historical data for all tickers from start_date to end_date using yfinance
    #data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True, progress=False)

    # Use this:
    data = get_yfinance_data(tickers, start_date, end_date)


    tickers_data = {}
    for ticker in tickers:
        try:
            df = data[ticker].copy()  # Copy data for the specific ticker
            df.reset_index(inplace=True)  # Reset the index so that Date becomes a column
            # Ensure the DataFrame contains the required columns before proceeding
            if not set(['Date','Open','High','Low','Close','Volume']).issubset(df.columns):
                continue
            df['UserAlpha'] = compute_user_alpha(df, alpha_formula)  # Calculate user-defined alpha values
            tickers_data[ticker] = df  # Store the processed DataFrame in a dictionary keyed by ticker
        except Exception as e:
            print(f"Error processing ticker {ticker}: {e}")

    if not tickers_data:
        print("No ticker data available.")
        return None, None  # Return early if no valid ticker data was processed

    # Merge alpha values from all tickers into one DataFrame
    alpha_list = []
    for ticker, df in tickers_data.items():
        temp = df[['Date', 'UserAlpha']].copy().set_index('Date')  # Extract Date and UserAlpha and set Date as index
        temp.rename(columns={'UserAlpha': ticker}, inplace=True)  # Rename the column to the ticker symbol
        alpha_list.append(temp)  # Add to the list for concatenation
    alpha_df = pd.concat(alpha_list, axis=1, join='outer').reset_index()  # Combine all ticker alphas into one DataFrame

    # Compute the daily cross-sectional threshold based on the specified percentile
    def daily_threshold(row):
        valid = row.dropna()  # Remove NaN values from the row
        # If there are valid numbers, compute the percentile; otherwise, return NaN
        return np.nan if valid.empty else np.nanpercentile(valid, cs_percentile * 100)
    # Apply the threshold calculation across all dates
    alpha_df['Threshold'] = alpha_df.drop(columns=['Date']).apply(daily_threshold, axis=1)

    # Merge the threshold into each ticker's DataFrame and generate trade signals
    for ticker, df in tickers_data.items():
        df = pd.merge(df, alpha_df[['Date','Threshold']], on='Date', how='left')
        df['Signal'] = df['UserAlpha'] > df['Threshold']  # Signal is True if alpha exceeds the threshold
        df['SignalShift'] = df['Signal'].shift(1)  # Shift the signal by one day to simulate trading on the next day
        tickers_data[ticker] = df

    all_trades = []
    # Simulate trade execution for each ticker based on shifted signals
    for ticker, df in tickers_data.items():
        trade_returns = []
        for i in range(len(df) - 1):
            if df.loc[i, 'SignalShift']:
                entry_date = df.loc[i+1, 'Date']  # Trade entry on the next day
                entry_price = df.loc[i+1, 'Open']  # Buy at the opening price
                exit_price = df.loc[i+1, 'Close']  # Sell at the closing price
                trade_ret = (exit_price - entry_price) / entry_price  # Compute the return of the trade
                trade_returns.append(trade_ret)
                # Record detailed trade information for later analysis
                all_trades.append({
                    'Ticker': ticker,
                    'EntryDate': entry_date,
                    'EntryPrice': entry_price,
                    'ExitDate': entry_date,
                    'ExitPrice': exit_price,
                    'TradeReturn': trade_ret
                })
            else:
                trade_returns.append(np.nan)  # No trade executed on this day
        df['TradeReturn'] = trade_returns + [np.nan]  # Append NaN for alignment (last day no trade)
        tickers_data[ticker] = df

    # Aggregate daily trade returns from all tickers into a portfolio-level DataFrame
    trade_returns_list = []
    for ticker, df in tickers_data.items():
        temp = df[['Date', 'TradeReturn']].copy().set_index('Date')  # Extract Date and TradeReturn
        temp.rename(columns={'TradeReturn': ticker}, inplace=True)  # Rename column to the ticker symbol
        trade_returns_list.append(temp)
    portfolio_df = pd.concat(trade_returns_list, axis=1, join='outer').reset_index()  # Combine returns across tickers
    portfolio_df.sort_values('Date', inplace=True)  # Sort the DataFrame by Date
    # Compute the portfolio return as the mean of all available ticker returns for each day
    portfolio_df['PortfolioReturn'] = portfolio_df.drop(columns=['Date']).mean(axis=1, skipna=True).fillna(0)
    # Calculate the cumulative equity curve from daily portfolio returns
    portfolio_df['Equity'] = (1 + portfolio_df['PortfolioReturn']).cumprod()

    trades_df = pd.DataFrame(all_trades)  # Create a DataFrame with details of every executed trade
    trades_df.sort_values('EntryDate', inplace=True)  # Sort trades chronologically by entry date
    trades_df.reset_index(drop=True, inplace=True)  # Reset index for neatness
    
    return portfolio_df, trades_df  # Return both the portfolio performance and trades DataFrames

#############################################
# Main Execution
#############################################
def main():
    print("Welcome to the Robust No-Code SP500 Portfolio Backtester with Performance Metrics!")
    print("This version uses cross-sectional filtering so that only stocks in the top percentile of alpha each day trade.")
    # Display introductory messages to the user

    user_alpha_formula = input(
        "Enter your alpha formula (e.g. (CLOSE-OPEN)/CLOSE or ts_rank(delta(((CLOSE-OPEN)/(HIGH-LOW))*VOLUME, 1), 5): "
    )
    # Prompt the user for a custom alpha formula
    if not user_alpha_formula.strip():
        user_alpha_formula = "(CLOSE-OPEN)/CLOSE"  # Default formula if the user provides no input
    
    cs_percentile_input = input("Enter cross-sectional percentile for trading (default 0.9 for top 10%): ")
    try:
        cs_percentile = float(cs_percentile_input) if cs_percentile_input.strip() else 0.9
        # Convert user input to a float; default to 0.9 if input is empty
    except ValueError:
        print("Invalid percentile input. Using default value 0.9.")
        cs_percentile = 0.9  # Fallback to default value if conversion fails

    start_date = input("Enter start date (YYYY-MM-DD, default 2000-01-01): ").strip() or "2000-01-01"
    # Get the start date from the user; default if blank
    end_date = input("Enter end date (YYYY-MM-DD, default 2025-01-01): ").strip() or "2025-01-01"
    # Get the end date from the user; default if blank

    print("\nUsing alpha formula:", user_alpha_formula)
    print("Using cross-sectional percentile filter:", cs_percentile)
    print(f"Running portfolio backtest for SP500 stocks from {start_date} to {end_date}...")
    # Echo back the parameters that will be used for the backtest

    portfolio_df, trades_df = portfolio_backtest_sp500(user_alpha_formula, cs_percentile, start_date, end_date)
    # Execute the backtest using the provided parameters
    if portfolio_df is None:
        print("Portfolio backtest failed.")
        return  # Exit if the backtest did not complete successfully

    total_return = portfolio_df['Equity'].iloc[-1] - 1.0  # Compute total return based on final equity value
    num_days = portfolio_df.shape[0]  # Count the total number of trading days in the backtest
    avg_daily_return = portfolio_df['PortfolioReturn'].mean()  # Calculate the average daily return
    aggregated_metrics = (
        f"Portfolio Performance Metrics (Aggregated):\n"
        f"Total Return: {total_return:.2%}\n"
        f"Number of Trading Days: {num_days}\n"
        f"Average Daily Return: {avg_daily_return:.4%}\n"
    )
    print("\n" + aggregated_metrics)
    # Print the aggregated performance metrics to the console

    tester = AlphaTester(portfolio_df['PortfolioReturn'])
    # Create an AlphaTester instance using the portfolio's daily returns
    metrics = tester.summary()  # Compute a full set of performance metrics
    detailed_metrics = "\nDetailed Performance Metrics:\n" + "\n".join(f"{k}: {v}" for k, v in metrics.items())
    print(detailed_metrics)
    # Print detailed metrics for further insight

    if not trades_df.empty:
        sample_trades = trades_df.head(10).to_string(index=False)
        trade_info = f"\nSample of Executed Trades:\n{sample_trades}\nTotal trades executed: {len(trades_df)}\n"
        print(trade_info)
        # If trades were executed, print a sample and the total count
    else:
        trade_info = "No trades executed based on the provided alpha and filtering criteria."
        print(trade_info)
        # Inform the user if no trades met the criteria

    # Save the backtest results and performance metrics to a text file
    with open("backtest_results.txt", "w") as f:
        f.write("Robust No-Code SP500 Portfolio Backtest Results\n")
        f.write("===============================================\n\n")
        f.write(aggregated_metrics + "\n")
        f.write(detailed_metrics + "\n\n")
        f.write(trade_info + "\n")
        f.write("Sample Portfolio Equity Curve Data (first 10 rows):\n")
        f.write(portfolio_df.head(10).to_string(index=False))
    print("\nResults have been saved to 'backtest_results.txt'.")
    
    # Plot the portfolio equity curve over time
    plt.figure(figsize=(12, 6))  # Set up the figure with a specified size
    dates = pd.to_datetime(portfolio_df['Date'])  # Convert the Date column to datetime objects for accurate plotting
    plt.plot(dates, portfolio_df['Equity'], label="Portfolio Equity", color="green")  # Plot the equity curve
    plt.xlabel("Date")  # Label the x-axis
    plt.ylabel("Equity")  # Label the y-axis
    plt.title("SP500 Portfolio Backtest Equity Curve")  # Provide a title for the plot
    plt.legend()  # Show the legend
    plt.show()  # Render the plot

if __name__ == "__main__":
    main()  # Run the main function if this script is executed directly
