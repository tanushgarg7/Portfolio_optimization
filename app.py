import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from deap import base, creator, tools, algorithms
import random

# Set the style for visualization
plt.style.use('fivethirtyeight')

# Title of the app
st.title('Gemstone Investment Portfolio Optimizer')

# Fetching real-time data
@st.cache_data
def load_data(tickers):
    df = pd.DataFrame()
    for ticker in tickers:
        df[ticker] = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
    return df

# Define assets and load data
assets = ['GC=F', 'SI=F', 'PL=F']  # Gold, Silver, Platinum
start_date = datetime(2013, 1, 1).strftime('%Y-%m-%d')
end_date = datetime.today().strftime('%Y-%m-%d')
df = load_data(assets)

# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# User Defined Portfolio Weights Input
st.header('Input Your Portfolio Weights')
user_weights = []
for asset in assets:
    weight = st.slider(f'Weight for {asset} (%)', 0, 100, 33)
    user_weights.append(weight / 100.0)
user_weights = np.array(user_weights)

# Displaying User Portfolio Performance
if st.button('Calculate Portfolio Performance'):
    ef = EfficientFrontier(mu, S)
    annual_return = np.dot(user_weights, mu)
    annual_volatility = np.sqrt(np.dot(user_weights.T, np.dot(S, user_weights)))
    sharpe_ratio = annual_return / annual_volatility
    st.subheader('Your Portfolio Performance:')
    st.write(f"Annualized Return: {annual_return:.2f}")
    st.write(f"Volatility: {annual_volatility:.2f}")
    st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Display Optimized Portfolios based on various optimization techniques
st.header('Optimized Portfolio Analysis')

# Maximum Sharpe Ratio Optimization
st.header('Optimized Portfolio for Maximum Sharpe Ratio')
ef = EfficientFrontier(mu, S)
sharpe_pwt = ef.max_sharpe()
sharpe_cleaned = ef.clean_weights()
sharpe_performance = ef.portfolio_performance(verbose=True)
st.subheader('Optimal Portfolio Weights:')
st.write(sharpe_cleaned)
st.subheader('Portfolio Performance:')
st.write(f"Expected annual return: {sharpe_performance[0]*100:.2f}%")
st.write(f"Annual volatility: {sharpe_performance[1]*100:.2f}%")
fig, ax = plt.subplots()
pd.Series(sharpe_cleaned).plot.pie(ax=ax, figsize=(10, 5), autopct='%1.1f%%')
st.pyplot(fig)

# Minimum Volatility Optimization
st.header('Optimized Portfolio for Minimum Volatility')
ef = EfficientFrontier(mu, S)
min_vol_pwt = ef.min_volatility()
min_vol_cleaned = ef.clean_weights()
min_vol_performance = ef.portfolio_performance(verbose=True)
st.subheader('Optimal Portfolio Weights:')
st.write(min_vol_cleaned)
st.subheader('Portfolio Performance:')
st.write(f"Expected annual return: {min_vol_performance[0]*100:.2f}%")
st.write(f"Annual volatility: {min_vol_performance[1]*100:.2f}%")
fig, ax = plt.subplots()
pd.Series(min_vol_cleaned).plot.pie(ax=ax, figsize=(10, 5), autopct='%1.1f%%')
st.pyplot(fig)

# Genetic Algorithm for Portfolio Optimization
st.header('Optimized Portfolio using Genetic Algorithm')
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(assets))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_portfolio(individual):
    returns = np.dot(individual, mu)
    std = np.sqrt(np.dot(individual, np.dot(S, individual)))
    sharpe_ratio = returns / std
    return sharpe_ratio,

toolbox.register("evaluate", eval_portfolio)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_genetic_algorithm():
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats, halloffame=hof, verbose=True)
    best_individual = hof.items[0]
    
    # Ensure non-negativity of weights and normalize
    best_individual = np.maximum(best_individual, 0)
    best_individual /= np.sum(best_individual)
    
    return best_individual

# Run genetic algorithm optimization
best_individual = run_genetic_algorithm()

# Display optimal weights
st.subheader('Optimal Portfolio Weights:')
for i, asset in enumerate(assets):
    st.write(f"{asset}: {best_individual[i]}")

# Calculate portfolio performance metrics
st.subheader('Portfolio Performance:')
portfolio_return = np.dot(best_individual, mu)
portfolio_std = np.sqrt(np.dot(best_individual, np.dot(S, best_individual)))

# Display portfolio performance metrics
st.write(f"Expected Annual Return: {portfolio_return*100:.2f}%")
st.write(f"Annual Volatality: {portfolio_std*100:.2f}%")
fig, ax = plt.subplots()
pd.Series(best_individual, index=assets).plot.pie(ax=ax, figsize=(10, 5), autopct='%1.1f%%')
st.pyplot(fig)

# Sidebar for user inputs
investment = st.sidebar.number_input("Enter investment amount($)", 10, 100000000, 1000)
objective = st.sidebar.radio("Select your investment objective", ["Maximize Sharpe Ratio", "Minimize Volatility", "Genetic Algorithm Optimization", "Maximize Return for Given Risk"])
max_risk = st.sidebar.slider("Maximum Risk Level (%)", 16, 50, 25) / 100.0 if objective == "Maximize Return for Given Risk" else None

# Discrete Allocation Section
st.header('Discrete Allocation Based on Your Objective')

latest_prices = get_latest_prices(df)

# Function to get weights based on objective
def get_weights(objective, max_risk=None):
    if objective == "Maximize Sharpe Ratio":
        return sharpe_cleaned
    elif objective == "Minimize Volatility":
        return min_vol_cleaned
    elif objective == "Genetic Algorithm Optimization":
        weights = dict(zip(assets, best_individual))
        return weights
    elif objective == "Maximize Return for Given Risk":
        ef = EfficientFrontier(mu, S)
        ef.efficient_risk(target_volatility=max_risk)
        return ef.clean_weights()

# Get the optimal weights based on the selected objective
optimal_weights = get_weights(objective, max_risk)

# Perform Discrete Allocation using the obtained weights
da = DiscreteAllocation(optimal_weights, latest_prices, total_portfolio_value=investment)
allocation, leftover = da.lp_portfolio()

# Display the results
st.write(f"Objective: {objective}")
st.write(f"Allocation: {allocation}")
st.write(f"Funds remaining: ${leftover:.2f}")