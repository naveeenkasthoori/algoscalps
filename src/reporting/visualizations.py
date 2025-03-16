"""
Visualization utilities for trading system reporting.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 12

def plot_equity_curve(equity_data: pd.Series, output_dir: str, filename: str = None) -> str:
    """
    Plot equity curve from backtest results.
    
    Args:
        equity_data: Series with equity values by date
        output_dir: Directory to save the plot
        filename: Optional filename, if None will generate a timestamped name
        
    Returns:
        Path to the saved plot file
    """
    if equity_data.empty:
        logger.warning("Empty equity data, cannot generate plot")
        return ""
    
    logger.info("Generating equity curve plot")
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"equity_curve_{timestamp}.png"
    
    # Create full path
    file_path = os.path.join(output_dir, filename)
    
    try:
        # Create figure
        plt.figure()
        
        # Plot equity curve
        plt.plot(equity_data.index, equity_data.values, 'b-', linewidth=2)
        
        # Add reference line at starting equity
        plt.axhline(y=equity_data.iloc[0], color='k', linestyle='--', alpha=0.5)
        
        # Add grid and labels
        plt.grid(True)
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity (₹)')
        
        # Format y-axis with commas for thousands
        plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
        
        # Add annotations for starting and ending equity
        plt.annotate(f'Start: ₹{equity_data.iloc[0]:,.0f}', 
                     xy=(equity_data.index[0], equity_data.iloc[0]),
                     xytext=(10, 10), textcoords='offset points')
        
        plt.annotate(f'End: ₹{equity_data.iloc[-1]:,.0f}', 
                     xy=(equity_data.index[-1], equity_data.iloc[-1]),
                     xytext=(-10, 10), textcoords='offset points',
                     horizontalalignment='right')
        
        # Save figure
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Equity curve plot saved to {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Error generating equity curve plot: {e}")
        return ""

def plot_drawdown_chart(equity_data: pd.Series, output_dir: str, filename: str = None) -> str:
    """
    Plot drawdown chart from equity curve.
    
    Args:
        equity_data: Series with equity values by date
        output_dir: Directory to save the plot
        filename: Optional filename, if None will generate a timestamped name
        
    Returns:
        Path to the saved plot file
    """
    if equity_data.empty:
        logger.warning("Empty equity data, cannot generate drawdown plot")
        return ""
    
    logger.info("Generating drawdown chart")
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drawdown_chart_{timestamp}.png"
    
    # Create full path
    file_path = os.path.join(output_dir, filename)
    
    try:
        # Calculate running maximum and drawdown
        running_max = equity_data.cummax()
        drawdown = (equity_data - running_max) / running_max * 100  # As percentage
        
        # Create figure
        plt.figure()
        
        # Plot drawdown
        plt.fill_between(drawdown.index, drawdown.values, 0, color='r', alpha=0.3)
        plt.plot(drawdown.index, drawdown.values, 'r-', linewidth=1)
        
        # Add zero line
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        
        # Add grid and labels
        plt.grid(True)
        plt.title('Drawdown Chart')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        
        # Invert y-axis as drawdowns are negative
        plt.gca().invert_yaxis()
        
        # Annotate maximum drawdown
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        plt.annotate(f'Max Drawdown: {max_dd:.1f}%', 
                     xy=(max_dd_date, max_dd),
                     xytext=(10, 30), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='k', alpha=0.5))
        
        # Save figure
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Drawdown chart saved to {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Error generating drawdown chart: {e}")
        return ""

def plot_pnl_distribution(trade_data: List[Dict[str, Any]], output_dir: str, filename: str = None) -> str:
    """
    Plot P&L distribution from trade data.
    
    Args:
        trade_data: List of trade dictionaries
        output_dir: Directory to save the plot
        filename: Optional filename, if None will generate a timestamped name
        
    Returns:
        Path to the saved plot file
    """
    if not trade_data:
        logger.warning("No trade data, cannot generate P&L distribution plot")
        return ""
    
    logger.info("Generating P&L distribution plot")
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pnl_distribution_{timestamp}.png"
    
    # Create full path
    file_path = os.path.join(output_dir, filename)
    
    try:
        # Extract P&L values from trade data
        pnl_values = [trade.get('total_pnl', trade.get('pnl', 0)) for trade in trade_data]
        
        # Create figure
        plt.figure()
        
        # Plot histogram with density curve
        sns.histplot(pnl_values, kde=True, bins=30)
        
        # Add vertical line at zero
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        
        # Add grid and labels
        plt.grid(True, alpha=0.3)
        plt.title('P&L Distribution')
        plt.xlabel('P&L (₹)')
        plt.ylabel('Frequency')
        
        # Format x-axis with commas for thousands
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
        
        # Add annotations for statistics
        mean_pnl = np.mean(pnl_values)
        median_pnl = np.median(pnl_values)
        std_pnl = np.std(pnl_values)
        
        stats_text = (f"Mean: ₹{mean_pnl:,.0f}\n"
                      f"Median: ₹{median_pnl:,.0f}\n"
                      f"Std Dev: ₹{std_pnl:,.0f}")
        
        # Position the text box in the upper right
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                     fontsize=10, ha='right', va='top', bbox=props)
        
        # Save figure
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"P&L distribution plot saved to {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Error generating P&L distribution plot: {e}")
        return ""

def plot_monthly_performance(equity_data: pd.Series, output_dir: str, filename: str = None) -> str:
    """
    Plot monthly performance heatmap.
    
    Args:
        equity_data: Series with equity values by date
        output_dir: Directory to save the plot
        filename: Optional filename, if None will generate a timestamped name
        
    Returns:
        Path to the saved plot file
    """
    if equity_data.empty:
        logger.warning("Empty equity data, cannot generate monthly performance plot")
        return ""
    
    logger.info("Generating monthly performance heatmap")
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"monthly_performance_{timestamp}.png"
    
    # Create full path
    file_path = os.path.join(output_dir, filename)
    
    try:
        # Calculate daily returns
        daily_returns = equity_data.pct_change().dropna()
        
        # Extract month and year
        daily_returns.index = pd.to_datetime(daily_returns.index)
        monthly_data = daily_returns.groupby([daily_returns.index.year, daily_returns.index.month]).sum()
        
        # Reshape into matrix for heatmap (year x month)
        years = sorted(set(daily_returns.index.year))
        months = range(1, 13)
        
        perf_matrix = np.zeros((len(years), 12))
        
        for i, year in enumerate(years):
            for j, month in enumerate(months):
                try:
                    perf_matrix[i, j] = monthly_data.loc[(year, month)]
                except:
                    perf_matrix[i, j] = np.nan
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Create heatmap
        ax = sns.heatmap(perf_matrix, cmap='RdYlGn', center=0,
                         xticklabels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                         yticklabels=years,
                         annot=True, fmt='.1%')
        
        # Add labels
        plt.title('Monthly Performance (%)')
        plt.xlabel('Month')
        plt.ylabel('Year')
        
        # Adjust annotation text color based on background
        for text in ax.texts:
            value = float(text.get_text()[:-1])  # Remove % and convert to float
            if abs(value) > 10:  # Threshold for changing text color
                text.set_color('white')
        
        # Save figure
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Monthly performance heatmap saved to {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Error generating monthly performance heatmap: {e}")
        return ""

def plot_trade_duration_vs_pnl(trade_data: List[Dict[str, Any]], output_dir: str, filename: str = None) -> str:
    """
    Plot trade duration vs. P&L scatter plot.
    
    Args:
        trade_data: List of trade dictionaries
        output_dir: Directory to save the plot
        filename: Optional filename, if None will generate a timestamped name
        
    Returns:
        Path to the saved plot file
    """
    if not trade_data:
        logger.warning("No trade data, cannot generate duration vs. P&L plot")
        return ""
    
    logger.info("Generating trade duration vs. P&L plot")
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"duration_vs_pnl_{timestamp}.png"
    
    # Create full path
    file_path = os.path.join(output_dir, filename)
    
    try:
        # Extract duration and P&L from trade data
        durations = []
        pnl_values = []
        option_types = []
        
        for trade in trade_data:
            duration = trade.get('duration_minutes', 0)
            pnl = trade.get('total_pnl', trade.get('pnl', 0))
            option_type = trade.get('option_type', 'Unknown')
            
            durations.append(duration)
            pnl_values.append(pnl)
            option_types.append(option_type)
        
        # Create DataFrame for plotting
        trade_df = pd.DataFrame({
            'Duration (min)': durations,
            'P&L': pnl_values,
            'Option Type': option_types
        })
        
        # Create figure
        plt.figure()
        
        # Plot scatter with different colors for calls and puts
        sns.scatterplot(data=trade_df, x='Duration (min)', y='P&L', 
                        hue='Option Type', palette={'CE': 'green', 'PE': 'red'},
                        alpha=0.7, s=70)
        
        # Add horizontal line at zero P&L
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Add grid and labels
        plt.grid(True, alpha=0.3)
        plt.title('Trade Duration vs. P&L')
        plt.xlabel('Duration (minutes)')
        plt.ylabel('P&L (₹)')
        
        # Format y-axis with commas for thousands
        plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
        
        # Add regression line
        if len(durations) > 2:
            x = np.array(durations)
            y = np.array(pnl_values)
            
            # Remove extreme outliers for better fit
            mask = (np.abs(x - np.mean(x)) < 3 * np.std(x)) & (np.abs(y - np.mean(y)) < 3 * np.std(y))
            x_filtered = x[mask]
            y_filtered = y[mask]
            
            if len(x_filtered) > 2:
                z = np.polyfit(x_filtered, y_filtered, 1)
                p = np.poly1d(z)
                
                # Plot trendline
                x_line = np.linspace(min(x), max(x), 100)
                plt.plot(x_line, p(x_line), 'k--', alpha=0.7)
                
                # Annotate correlation
                corr = np.corrcoef(x_filtered, y_filtered)[0, 1]
                plt.annotate(f'Correlation: {corr:.2f}', xy=(0.95, 0.05), 
                             xycoords='axes fraction', fontsize=10, ha='right', va='bottom',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Save figure
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Trade duration vs. P&L plot saved to {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Error generating trade duration vs. P&L plot: {e}")
        return ""

def plot_exit_reasons_pie(trade_data: List[Dict[str, Any]], output_dir: str, filename: str = None) -> str:
    """
    Plot pie chart of trade exit reasons.
    
    Args:
        trade_data: List of trade dictionaries
        output_dir: Directory to save the plot
        filename: Optional filename, if None will generate a timestamped name
        
    Returns:
        Path to the saved plot file
    """
    if not trade_data:
        logger.warning("No trade data, cannot generate exit reasons pie chart")
        return ""
    
    logger.info("Generating exit reasons pie chart")
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exit_reasons_pie_{timestamp}.png"
    
    # Create full path
    file_path = os.path.join(output_dir, filename)
    
    try:
        # Extract exit reasons from trade data
        exit_reasons = [trade.get('exit_reason', 'unknown') for trade in trade_data]
        
        # Count occurrences of each reason
        reason_counts = {}
        for reason in exit_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        # Sort reasons by count (descending)
        sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare data for pie chart
        labels = [item[0] for item in sorted_reasons]
        sizes = [item[1] for item in sorted_reasons]
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Generate colors
        colors = plt.cm.tab10(range(len(labels)))
        
        # Create pie chart
        patches, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%',
                                           startangle=90, colors=colors, shadow=False)
        
        # Make percentage text easier to read
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
        
        # Equal aspect ratio ensures pie is circular
        plt.axis('equal')
        
        # Add title
        plt.title('Trade Exit Reasons')
        
        # Add legend
        plt.legend(labels, title="Exit Reasons", 
                   loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        # Save figure
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Exit reasons pie chart saved to {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Error generating exit reasons pie chart: {e}")
        return ""

def plot_daily_pnl_bar(daily_pnl: Dict[datetime.date, float], output_dir: str, filename: str = None) -> str:
    """
    Plot daily P&L bar chart.
    
    Args:
        daily_pnl: Dictionary with daily P&L by date
        output_dir: Directory to save the plot
        filename: Optional filename, if None will generate a timestamped name
        
    Returns:
        Path to the saved plot file
    """
    if not daily_pnl:
        logger.warning("No daily P&L data, cannot generate bar chart")
        return ""
    
    logger.info("Generating daily P&L bar chart")
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"daily_pnl_bar_{timestamp}.png"
    
    # Create full path
    file_path = os.path.join(output_dir, filename)
    
    try:
        # Convert dictionary to Series
        daily_series = pd.Series(daily_pnl)
        daily_series.index = pd.to_datetime(daily_series.index)
        daily_series = daily_series.sort_index()
        
        # Calculate cumulative P&L
        cumulative = daily_series.cumsum()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1]})
        
        # Plot daily P&L bars
        colors = ['green' if x > 0 else 'red' for x in daily_series.values]
        ax1.bar(daily_series.index, daily_series.values, color=colors, alpha=0.7)
        
        # Add zero line
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        
        # Add grid and labels for daily P&L
        ax1.grid(True, axis='y', alpha=0.3)
        ax1.set_title('Daily P&L')
        ax1.set_xlabel('')
        ax1.set_ylabel('P&L (₹)')
        
        # Format y-axis with commas for thousands
        ax1.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
        
        # Annotate best and worst days
        best_day = daily_series.idxmax()
        worst_day = daily_series.idxmin()
        
        ax1.annotate(f'Best: ₹{daily_series.max():,.0f}', 
                    xy=(best_day, daily_series.max()),
                    xytext=(0, 20), textcoords='offset points',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', fc='green', alpha=0.3))
        
        ax1.annotate(f'Worst: ₹{daily_series.min():,.0f}', 
                    xy=(worst_day, daily_series.min()),
                    xytext=(0, -20), textcoords='offset points',
                    ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='red', alpha=0.3))
        
        # Plot cumulative P&L line
        ax2.plot(cumulative.index, cumulative.values, 'b-', linewidth=2)
        
        # Add zero line
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        
        # Add grid and labels for cumulative P&L
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Cumulative P&L')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulative P&L (₹)')
        
        # Format y-axis with commas for thousands
        ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Daily P&L bar chart saved to {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Error generating daily P&L bar chart: {e}")
        return ""

def create_all_charts(backtest_results: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """
    Create all available charts from backtest results.
    
    Args:
        backtest_results: Dictionary with backtest results
        output_dir: Directory to save the plots
        
    Returns:
        Dictionary mapping chart names to file paths
    """
    logger.info("Creating all charts from backtest results")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary
    chart_paths = {}
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract data from backtest results
    equity_curve = backtest_results.get('equity_curve', pd.Series())
    trades = backtest_results.get('trades', [])
    daily_pnl = backtest_results.get('daily_pnl', {})
    
    # Create equity curve plot
    if not equity_curve.empty:
        equity_file = f"equity_curve_{timestamp}.png"
        path = plot_equity_curve(equity_curve, output_dir, equity_file)
        if path:
            chart_paths['equity_curve'] = path
    
    # Create drawdown chart
    if not equity_curve.empty:
        drawdown_file = f"drawdown_chart_{timestamp}.png"
        path = plot_drawdown_chart(equity_curve, output_dir, drawdown_file)
        if path:
            chart_paths['drawdown_chart'] = path
    
    # Create P&L distribution plot
    if trades:
        pnl_dist_file = f"pnl_distribution_{timestamp}.png"
        path = plot_pnl_distribution(trades, output_dir, pnl_dist_file)
        if path:
            chart_paths['pnl_distribution'] = path
    
    # Create monthly performance heatmap
    if not equity_curve.empty:
        monthly_file = f"monthly_performance_{timestamp}.png"
        path = plot_monthly_performance(equity_curve, output_dir, monthly_file)
        if path:
            chart_paths['monthly_performance'] = path
    
    # Create trade duration vs. P&L plot
    if trades:
        duration_file = f"duration_vs_pnl_{timestamp}.png"
        path = plot_trade_duration_vs_pnl(trades, output_dir, duration_file)
        if path:
            chart_paths['duration_vs_pnl'] = path
    
    # Create exit reasons pie chart
    if trades:
        exit_file = f"exit_reasons_pie_{timestamp}.png"
        path = plot_exit_reasons_pie(trades, output_dir, exit_file)
        if path:
            chart_paths['exit_reasons_pie'] = path
    
    # Create daily P&L bar chart
    if daily_pnl:
        daily_file = f"daily_pnl_bar_{timestamp}.png"
        path = plot_daily_pnl_bar(daily_pnl, output_dir, daily_file)
        if path:
            chart_paths['daily_pnl_bar'] = path
    
    logger.info(f"Created {len(chart_paths)} charts from backtest results")
    return chart_paths