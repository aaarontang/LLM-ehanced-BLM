"""
Visualization Module (GitHub LLM-BLM Style)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import glob
import matplotlib.ticker as mtick
import matplotlib
matplotlib.use('Agg')

# --- STYLE SETTINGS ---
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("deep")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['figure.dpi'] = 300

def save_plot(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved Plot: {path}")

# --- PLOT 1: WEALTH INDEX ---
def plot_cumulative_wealth(strategies_returns, dates, save_path='results/1_wealth_index.png'):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # GitHub Project Colors
    colors = {
        'LLM-BL': '#756bb1',  # Purple
        'MVO': '#d95f02',     # Orange/Red
        'EW': '#1b9e77',      # Green
        'Market': '#666666'   # Grey
    }
    
    for name, returns in strategies_returns.items():
        n = min(len(returns), len(dates))
        wealth_index = np.cumprod(1 + np.array(returns[:n]))
        wealth_index = np.insert(wealth_index, 0, 1.0)
        plot_dates = np.insert(dates[:n], 0, dates[0] - pd.DateOffset(days=1))
        
        style = {'linewidth': 2.5} if name == 'LLM-BL' else {'linewidth': 2.0, 'linestyle': '--'}
        ax.plot(plot_dates, wealth_index, label=name, color=colors.get(name, 'blue'), **style)
    
    ax.set_title('Cumulative Return', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(title='Portfolio', loc='upper left', frameon=True, framealpha=0.9)
    
    # Grid and Formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    
    save_plot(fig, save_path)

# --- PLOT 2: STACKED WEIGHTS BAR CHART ---
def plot_weights_stacked_bar(weights_dict, tickers, strategy_name="LLM-BL", save_path='results/3_weights_bars.png'):
    sorted_dates = sorted(weights_dict.keys())
    if not sorted_dates: return

    n_dates = len(sorted_dates)
    n_tickers = len(tickers)
    
    # Prepare Data Matrix
    weight_matrix = np.zeros((n_tickers, n_dates))
    for i, date in enumerate(sorted_dates):
        w = weights_dict[date]
        if len(w) == n_tickers:
            weight_matrix[:, i] = w

    fig, ax = plt.subplots(figsize=(15, 8))
    
    # High contrast pastel colors
    cmap = plt.get_cmap('Pastel1')
    colors = [cmap(i % 9) for i in range(n_tickers)]
    
    bottoms = np.zeros(n_dates)
    date_labels = [d.strftime('%Y-%m-%d') for d in sorted_dates]
    x_pos = np.arange(n_dates)
    
    # Plot Bars
    for i in range(n_tickers):
        ax.bar(x_pos, weight_matrix[i], bottom=bottoms, label=tickers[i], 
               color=colors[i], edgecolor='white', width=0.85)
        
        # Add Labels inside bars (if weight is large enough)
        for j in range(n_dates):
            weight = weight_matrix[i, j]
            if weight > 0.04: # Only label if > 4%
                y_center = bottoms[j] + weight / 2
                ticker_short = tickers[i].replace('.HK', '')
                ax.text(x_pos[j], y_center, ticker_short, ha='center', va='center', 
                        fontsize=8, color='black', alpha=0.7)
        
        bottoms += weight_matrix[i]

    ax.set_title(f'{strategy_name} Weights Allocation', fontsize=16, fontweight='bold')
    ax.set_ylabel('Allocation', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(date_labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.margins(x=0.01)
    
    # Clean layout without legend (labels are inside bars)
    save_plot(fig, save_path)

# --- PLOT 3: VIEW DISTRIBUTION BOXPLOT ---
def plot_view_distribution(view_folder='results/views', save_path='results/2_view_boxplot.png'):
    """
    Reads all CSVs in results/views/ and plots the distribution of Q (Expected Returns).
    """
    view_files = glob.glob(os.path.join(view_folder, 'view_*.csv'))
    if not view_files:
        print("  ! No view files found for Box Plot.")
        return

    all_views = []
    for f in view_files:
        # Filename format: view_20240401.csv
        date_str = f.split('_')[-1].replace('.csv', '')
        date_obj = pd.to_datetime(date_str)
        
        df = pd.read_csv(f)
        if 'Q' in df.columns:
            temp = pd.DataFrame({
                'Date': date_obj,
                'Expected Return (%)': df['Q'] * 100 # Convert to %
            })
            all_views.append(temp)
            
    if not all_views: return
    
    big_df = pd.concat(all_views).sort_values('Date')
    big_df['Date_Str'] = big_df['Date'].dt.strftime('%Y-%m')

    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Boxplot
    sns.boxplot(x='Date_Str', y='Expected Return (%)', data=big_df, ax=ax,
                color='#bcbd22', width=0.5, fliersize=3, linewidth=1)
    
    ax.set_title('LLM Expected Returns Distribution (Views)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Rebalancing Date', fontsize=12)
    ax.set_ylabel('Expected Return (%)', fontsize=12)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.xticks(rotation=45)
    
    save_plot(fig, save_path)

# --- HELPER: STANDARD METRICS TABLE ---
def create_comparison_table(metrics_dict, save_path='results/5_performance_table.png'):
    df = pd.DataFrame(metrics_dict).T
    cols_map = {
        'TotalReturn': 'Total Return', 'CAGR': 'Ann. Return',
        'Sharpe': 'Sharpe Ratio', 'Volatility': 'Ann. Volatility',
        'MaxDrawdown': 'Max Drawdown'
    }
    df = df[list(cols_map.keys())].rename(columns=cols_map)
    
    cell_text = []
    for idx, row in df.iterrows():
        formatted = [f"{row[c]:.3f}" if 'Sharpe' in c else f"{row[c]*100:.2f}%" for c in df.columns]
        cell_text.append(formatted)
        
    fig, ax = plt.subplots(figsize=(10, len(df)*0.5+1))
    ax.axis('off')
    table = ax.table(cellText=cell_text, rowLabels=df.index, colLabels=df.columns, loc='center', cellLoc='center')
    table.scale(1, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    for j in range(len(df.columns)):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', weight='bold')
        
    save_plot(fig, save_path)
    df.to_csv(save_path.replace('.png', '.csv'))