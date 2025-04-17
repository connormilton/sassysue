#!/usr/bin/env python3
"""
Analysis tools for the Self-Evolving Forex Trading System
--------------------------------------------------------
This module provides tools for visualizing and analyzing
the performance of the trading system.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

def load_results(filename="trading_results.json"):
    """Load simulation results from a JSON file"""
    with open(filename, "r") as f:
        return json.load(f)

def plot_balance_history(results):
    """Plot the account balance over time"""
    balance_history = results["performance_history"]["balance_history"]
    
    if not balance_history:
        print("No balance history data available")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(balance_history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    
    # Plot balance history
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["balance"])
    plt.title("Account Balance Over Time")
    plt.xlabel("Date")
    plt.ylabel("Balance ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("balance_history.png")
    plt.close()
    
    print(f"Balance history plot saved as balance_history.png")
    
    return df

def plot_daily_returns(results):
    """Plot daily returns with target line"""
    daily_reviews = results["performance_history"]["daily_reviews"]
    
    if not daily_reviews:
        print("No daily review data available")
        return
    
    # Extract dates and returns
    dates = [datetime.strptime(dr["date"], "%Y-%m-%d") for dr in daily_reviews]
    returns = [dr["performance"]["daily_profit_pct"] for dr in daily_reviews]
    target = results["performance_history"]["daily_reviews"][0]["performance"]["target_daily_return"]
    
    # Plot daily returns
    plt.figure(figsize=(12, 6))
    plt.bar(dates, returns, color="blue", alpha=0.7)
    plt.axhline(y=target, color="r", linestyle="--", label=f"{target}% Target")
    plt.title("Daily Returns vs Target")
    plt.xlabel("Date")
    plt.ylabel("Daily Return (%)")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig("daily_returns.png")
    plt.close()
    
    print(f"Daily returns plot saved as daily_returns.png")
    
    # Calculate statistics
    hit_rate = sum(1 for r in returns if r >= target) / len(returns)
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    
    print(f"Target hit rate: {hit_rate:.2%}")
    print(f"Average daily return: {avg_return:.2f}%")
    print(f"Standard deviation: {std_return:.2f}%")
    print(f"Coefficient of variation: {std_return/avg_return:.2f}")
    
    return dates, returns

def analyze_strategy_performance(results):
    """Analyze and plot strategy performance"""
    trade_history = results["performance_history"]["trade_history"]
    
    if not trade_history:
        print("No trade history data available")
        return
    
    # Group trades by strategy
    strategies = {}
    
    for trade in trade_history:
        strategy = trade["strategy"]
        if strategy not in strategies:
            strategies[strategy] = []
        strategies[strategy].append(trade)
    
    # Calculate strategy statistics
    stats = {}
    
    for strategy, trades in strategies.items():
        wins = sum(1 for t in trades if t["profit_loss"] > 0)
        losses = sum(1 for t in trades if t["profit_loss"] <= 0)
        win_rate = wins / len(trades) if trades else 0
        
        avg_win = np.mean([t["profit_loss_pct"] for t in trades if t["profit_loss"] > 0]) if wins else 0
        avg_loss = np.mean([abs(t["profit_loss_pct"]) for t in trades if t["profit_loss"] <= 0]) if losses else 0
        
        stats[strategy] = {
            "trades": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": avg_win / avg_loss if avg_loss > 0 else float('inf'),
            "total_profit_pct": sum(t["profit_loss_pct"] for t in trades)
        }
    
    # Print strategy statistics
    print("\nStrategy Performance:")
    print("=====================")
    
    for strategy, stat in stats.items():
        print(f"\n{strategy}:")
        print(f"  Trades: {stat['trades']}")
        print(f"  Win Rate: {stat['win_rate']:.2%}")
        print(f"  Avg Win: {stat['avg_win']:.2f}%")
        print(f"  Avg Loss: {stat['avg_loss']:.2f}%")
        print(f"  Profit Factor: {stat['profit_factor']:.2f}")
        print(f"  Total Profit: {stat['total_profit_pct']:.2f}%")
    
    # Plot win rates
    plt.figure(figsize=(10, 6))
    strategies_list = list(stats.keys())
    win_rates = [stats[s]["win_rate"] for s in strategies_list]
    
    plt.bar(strategies_list, win_rates, color="green", alpha=0.7)
    plt.title("Strategy Win Rates")
    plt.xlabel("Strategy")
    plt.ylabel("Win Rate")
    plt.ylim(0, 1)
    for i, v in enumerate(win_rates):
        plt.text(i, v + 0.02, f"{v:.2%}", ha='center')
    plt.tight_layout()
    plt.savefig("strategy_win_rates.png")
    plt.close()
    
    # Plot profit by strategy
    plt.figure(figsize=(10, 6))
    profits = [stats[s]["total_profit_pct"] for s in strategies_list]
    
    plt.bar(strategies_list, profits, color="blue", alpha=0.7)
    plt.title("Total Profit by Strategy")
    plt.xlabel("Strategy")
    plt.ylabel("Total Profit (%)")
    for i, v in enumerate(profits):
        plt.text(i, v + (1 if v > 0 else -3), f"{v:.2f}%", ha='center')
    plt.tight_layout()
    plt.savefig("strategy_profits.png")
    plt.close()
    
    print("\nStrategy performance plots saved as strategy_win_rates.png and strategy_profits.png")
    
    return stats

def analyze_market_conditions(results):
    """Analyze performance in different market conditions"""
    trade_history = results["performance_history"]["trade_history"]
    
    if not trade_history:
        print("No trade history data available")
        return
    
    # Group trades by market condition
    conditions = {}
    
    for trade in trade_history:
        # Extract condition from market context
        context = trade.get("market_context", "")
        condition = None
        
        if "Market condition:" in context:
            condition = context.split("Market condition:")[1].strip()
            
        if condition:
            if condition not in conditions:
                conditions[condition] = []
            conditions[condition].append(trade)
    
    # Calculate condition statistics
    stats = {}
    
    for condition, trades in conditions.items():
        wins = sum(1 for t in trades if t["profit_loss"] > 0)
        losses = sum(1 for t in trades if t["profit_loss"] <= 0)
        win_rate = wins / len(trades) if trades else 0
        
        avg_win = np.mean([t["profit_loss_pct"] for t in trades if t["profit_loss"] > 0]) if wins else 0
        avg_loss = np.mean([abs(t["profit_loss_pct"]) for t in trades if t["profit_loss"] <= 0]) if losses else 0
        
        stats[condition] = {
            "trades": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": avg_win / avg_loss if avg_loss > 0 else float('inf'),
            "total_profit_pct": sum(t["profit_loss_pct"] for t in trades)
        }
    
    # Print condition statistics
    print("\nMarket Condition Performance:")
    print("============================")
    
    for condition, stat in stats.items():
        print(f"\n{condition}:")
        print(f"  Trades: {stat['trades']}")
        print(f"  Win Rate: {stat['win_rate']:.2%}")
        print(f"  Avg Win: {stat['avg_win']:.2f}%")
        print(f"  Avg Loss: {stat['avg_loss']:.2f}%")
        print(f"  Profit Factor: {stat['profit_factor']:.2f}")
        print(f"  Total Profit: {stat['total_profit_pct']:.2f}%")
    
    # Plot win rates by condition
    plt.figure(figsize=(12, 6))
    conditions_list = list(stats.keys())
    win_rates = [stats[c]["win_rate"] for c in conditions_list]
    
    plt.bar(conditions_list, win_rates, color="green", alpha=0.7)
    plt.title("Win Rates by Market Condition")
    plt.xlabel("Market Condition")
    plt.ylabel("Win Rate")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    for i, v in enumerate(win_rates):
        plt.text(i, v + 0.02, f"{v:.2%}", ha='center')
    plt.tight_layout()
    plt.savefig("condition_win_rates.png")
    plt.close()
    
    # Plot profit by condition
    plt.figure(figsize=(12, 6))
    profits = [stats[c]["total_profit_pct"] for c in conditions_list]
    
    plt.bar(conditions_list, profits, color="blue", alpha=0.7)
    plt.title("Total Profit by Market Condition")
    plt.xlabel("Market Condition")
    plt.ylabel("Total Profit (%)")
    plt.xticks(rotation=45, ha="right")
    for i, v in enumerate(profits):
        plt.text(i, v + (1 if v > 0 else -3), f"{v:.2f}%", ha='center')
    plt.tight_layout()
    plt.savefig("condition_profits.png")
    plt.close()
    
    print("\nMarket condition plots saved as condition_win_rates.png and condition_profits.png")
    
    return stats

def analyze_evolution(results):
    """Analyze how the system evolved over time"""
    prompt_versions = results.get("prompt_versions", [])
    
    if not prompt_versions:
        print("No evolution data available")
        return
    
    # Print evolution history
    print("\nSystem Evolution History:")
    print("========================")
    
    for i, version in enumerate(prompt_versions):
        print(f"\nVersion {i+1} - {version['timestamp']}")
        print(f"Reason: {version['reason']}")
        print(f"Changes: {version['prompt']}")
    
    return prompt_versions

def generate_summary_report(results):
    """Generate a comprehensive performance summary"""
    # Create the report text
    report = []
    report.append("=" * 50)
    report.append("Self-Evolving Forex Trading System - Performance Report")
    report.append("=" * 50)
    report.append("")
    
    # Overall performance
    report.append("Overall Performance:")
    report.append("-" * 20)
    report.append(f"Initial Balance: ${10000:.2f}")
    report.append(f"Final Balance: ${results['final_balance']:.2f}")
    report.append(f"Total Profit: ${results['total_profit']:.2f}")
    report.append(f"Total Return: {results['total_return_pct']:.2f}%")
    
    # Calculate annualized return
    days_run = 30  # Assuming 30 days
    annualized_return = ((1 + results['total_return_pct']/100) ** (365/days_run) - 1) * 100
    report.append(f"Annualized Return: {annualized_return:.2f}%")
    report.append("")
    
    # Trade statistics
    trade_history = results["performance_history"]["trade_history"]
    total_trades = len(trade_history)
    winning_trades = sum(1 for t in trade_history if t["profit_loss"] > 0)
    losing_trades = total_trades - winning_trades
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    report.append("Trade Statistics:")
    report.append("-" * 20)
    report.append(f"Total Trades: {total_trades}")
    report.append(f"Winning Trades: {winning_trades} ({win_rate:.2%})")
    report.append(f"Losing Trades: {losing_trades} ({1-win_rate:.2%})")
    
    if winning_trades > 0 and losing_trades > 0:
        avg_win = np.mean([t["profit_loss_pct"] for t in trade_history if t["profit_loss"] > 0])
        avg_loss = np.mean([abs(t["profit_loss_pct"]) for t in trade_history if t["profit_loss"] <= 0])
        report.append(f"Average Win: {avg_win:.2f}%")
        report.append(f"Average Loss: {avg_loss:.2f}%")
        report.append(f"Profit Factor: {avg_win/avg_loss:.2f}")
    report.append("")
    
    # Strategy summary
    strategy_counts = {}
    for trade in trade_history:
        strategy = trade["strategy"]
        if strategy not in strategy_counts:
            strategy_counts[strategy] = {"count": 0, "wins": 0, "profit": 0}
        
        strategy_counts[strategy]["count"] += 1
        if trade["profit_loss"] > 0:
            strategy_counts[strategy]["wins"] += 1
        strategy_counts[strategy]["profit"] += trade["profit_loss_pct"]
    
    report.append("Strategy Summary:")
    report.append("-" * 20)
    for strategy, stats in strategy_counts.items():
        win_rate = stats["wins"] / stats["count"] if stats["count"] > 0 else 0
        report.append(f"{strategy}: {stats['count']} trades, {win_rate:.2%} win rate, {stats['profit']:.2f}% total profit")
    report.append("")
    
    # Evolution summary
    prompt_versions = results.get("prompt_versions", [])
    report.append("Evolution Summary:")
    report.append("-" * 20)
    report.append(f"Total Evolutionary Changes: {len(prompt_versions)}")
    if prompt_versions:
        report.append("Key Evolutionary Moments:")
        for i, version in enumerate(prompt_versions[:min(5, len(prompt_versions))]):
            report.append(f"  - {version['reason']}")
    report.append("")
    
    # Save the report
    with open("performance_summary.txt", "w") as f:
        f.write("\n".join(report))
    
    print("\nPerformance summary saved as performance_summary.txt")
    
    return report

def main():
    """Run all analysis functions"""
    print("=" * 50)
    print("Self-Evolving Forex Trading System Analysis")
    print("=" * 50)
    
    # Check if results file exists
    if not os.path.exists("trading_results.json"):
        print("No trading_results.json file found. Please run the trading system first.")
        return
    
    # Load results
    results = load_results()
    
    # Run all analyses
    print("\nAnalyzing balance history...")
    plot_balance_history(results)
    
    print("\nAnalyzing daily returns...")
    plot_daily_returns(results)
    
    print("\nAnalyzing strategy performance...")
    analyze_strategy_performance(results)
    
    print("\nAnalyzing market conditions...")
    analyze_market_conditions(results)
    
    print("\nAnalyzing system evolution...")
    analyze_evolution(results)
    
    print("\nGenerating summary report...")
    generate_summary_report(results)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()