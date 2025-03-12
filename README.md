# Robust No-Code SP500 Portfolio Backtester with Performance Metrics

## Overview

This project is a no-code backtesting engine for S&P 500 stocks. It lets you evaluate custom alpha strategies using historical OHLCV data from yfinance. By computing performance metrics, simulating trades based on your own alpha formula, and saving results, this tool turns your ideas into actionable insights—all while offering feedback to help you learn from each step.

## Features

- **Data Acquisition & Caching:**  
  - Retrieves the list of S&P 500 companies from Wikipedia.
  - Downloads daily OHLCV data (Open, High, Low, Close, Volume) for each ticker via yfinance.
  - Caches data locally to save time on subsequent runs.

- **Custom Alpha Calculation:**  
  - Input your custom alpha formula (e.g., `(CLOSE-OPEN)/CLOSE` or more complex expressions).
  - Preprocesses formulas to standardize variable names.
  - Safely evaluates your formula to compute a unique alpha value for each stock.

- **Cross-Sectional Filtering & Signal Generation:**  
  - Applies a cross-sectional filter (e.g., top 10% of stocks by alpha) on a daily basis.
  - Generates trading signals when a stock's alpha exceeds the computed threshold.

- **Trade Simulation:**  
  - Simulates trades by executing a buy at the next day’s open and a sell at the close.
  - Records detailed trade information including entry/exit dates, prices, and returns.

- **Portfolio Aggregation & Performance Metrics:**  
  - Aggregates trade returns to form a daily portfolio return.
  - Constructs an equity curve by compounding daily returns.
  - Computes a suite of performance metrics such as:
    - Annualized Return
    - Annualized Volatility
    - Sharpe Ratio
    - Maximum Drawdown
    - Statistical tests (T-test, Wilcoxon test, Sign test)
    - Confidence Intervals
    - Monte Carlo Permutation Test

- **Output & Visualization:**  
  - Prints both aggregated and detailed performance metrics.
  - Saves results and sample trade details to `backtest_results.txt`.
  - Plots the portfolio equity curve for visual analysis.

## Dependencies

- Python 3.x
- [yfinance](https://pypi.org/project/yfinance/)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [scipy](https://www.scipy.org/)

Install the necessary packages using pip:

```bash
pip install yfinance numpy pandas matplotlib scipy
```

## How It Works

1. **Data Acquisition:**
   - The script fetches the list of S&P 500 companies from Wikipedia.
   - Historical OHLCV data is downloaded using yfinance.
   - A caching mechanism is in place (`sp500_yf_data.pkl`) to avoid redundant downloads for overlapping date ranges.

2. **Alpha Computation:**
   - You are prompted to enter a custom alpha formula.
   - The formula is preprocessed (e.g., converting variable names to uppercase) to maintain consistency.
   - It is then safely evaluated within a restricted environment to compute the alpha for each stock.

3. **Signal Generation & Trade Simulation:**
   - For each day, a cross-sectional threshold is computed based on the chosen percentile (default is 0.9, representing the top 10%).
   - If a stock’s alpha exceeds this threshold, a trade signal is generated.
   - Trades are simulated by buying at the open and selling at the close on the following day, and each trade’s details are recorded.

4. **Portfolio Aggregation & Metrics Calculation:**
   - Individual trade returns are aggregated into a portfolio-level daily return.
   - An equity curve is constructed by compounding these returns.
   - The `AlphaTester` class is used to compute performance metrics, providing insights into return, volatility, drawdowns, and statistical tests.

5. **Output & Visualization:**
   - Summary and detailed metrics are displayed on the console.
   - A text file (`backtest_results.txt`) is created with a comprehensive summary of the backtest.
   - A matplotlib plot visualizes the portfolio equity curve.

## How to Run

1. **Setup:**  
   Ensure you have Python 3.x installed and all dependencies installed via pip.

2. **Run the Script:**  
   Execute the script from the command line:

   ```bash
   python no_code_alpha_backtester.py
   ```

3. **Interactive Prompts:**  
   - **Alpha Formula:** Enter your custom formula (e.g., `(CLOSE-OPEN)/CLOSE`). A default is provided if left blank.
   - **Cross-Sectional Percentile:** Specify the percentile for filtering (default is 0.9 for top 10%).
   - **Date Range:** Enter the start and end dates (defaults are `2000-01-01` and `2025-01-01`).

4. **Review Results:**  
   - Check the console for a summary of performance metrics.
   - View the detailed results saved in `backtest_results.txt`.
   - Analyze the plotted equity curve to understand portfolio performance.

## Notes & Disclaimer

- **Risk Warning:**  
  Trading involves significant risk. This tool is provided for educational and research purposes only—use it at your own risk.

- **Caching:**  
  The script caches downloaded data in `sp500_yf_data.pkl`. To force a data update, modify the `force_update` parameter in the `get_yfinance_data` function.

- **Customization:**  
  Feel free to tweak the alpha formula and other parameters to fit your strategy. The code is heavily commented to help you understand each part.

- **Learning Approach:**  
  Remember, every execution is a learning experience. Use the feedback from each test run to refine your strategy. Let the process guide you toward smarter, bolder actions.

## Final Thoughts

This backtester is not just a script—it's a platform for turning insights into action. Every challenge you encounter is an opportunity to learn and grow. Trust the process, make adjustments, and know that every step forward brings you closer to mastering the craft. Happy backtesting and may your next bold move lead to remarkable breakthroughs!
