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

# Define color scheme for each asset
asset_colors = {
    'GC=F': 'gold',  # Gold
    'SI=F': 'silver',  # Silver
    'PL=F': 'deepskyblue'  # Platinum
}

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

# Choice of input method
input_method = st.radio("Choose your input method:",
                        ('Enter investment amounts', 'Input portfolio weights'))

if input_method == 'Enter investment amounts':
    # Input monetary investments
    st.header('Input Your Investment in Each Asset')
    investments = {}
    total_investment = 0
    latest_prices = get_latest_prices(df)
    for asset in assets:
        amount = st.number_input(f'Amount invested in {asset} ($)', min_value=0.0, max_value=1e9, value=0.0, step=1.0)
        investments[asset] = amount
        total_investment += amount

    # Compute weights based on investments and latest prices
    if total_investment > 0:
        weights = np.array([investments[asset] / latest_prices[asset] for asset in assets])
        weights /= np.sum(weights)  # Normalize to sum to 1
else:
    # User Defined Portfolio Weights Input
    st.header('Input Your Portfolio Weights')
    user_weights = []
    for asset in assets:
        weight = st.slider(f'Weight for {asset} (%)', 0, 100, 0)
        user_weights.append(weight / 100.0)
    user_weights = np.array(user_weights)
    weights = user_weights

# Displaying User Portfolio Performance
if st.button('Calculate Your Portfolio Performance'):
    # Check if weights sum exactly to 1 (or 100%)
    if np.sum(weights) != 1:
        st.error('The sum of weights must exactly equal 100%. Please adjust your weights.')
    else:
        ef = EfficientFrontier(mu, S)
        annual_return = np.dot(weights, mu)
        annual_volatility = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
        sharpe_ratio = annual_return / annual_volatility
        st.subheader('Your Portfolio Performance:')
        st.write(f"Annualized Return: {annual_return*100:.2f}%")
        st.write(f"Volatility: {annual_volatility*100:.2f}%")
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
# Filter out assets with zero weights to avoid clutter in the pie chart
filtered_weights = {k: v for k, v in sharpe_cleaned.items() if v > 0}
labels = list(filtered_weights.keys())
sizes = [filtered_weights[asset] for asset in labels]
colors = [asset_colors.get(asset, 'gray') for asset in labels]  # Assign colors
# Creating the pie chart
fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct=lambda pct: "{:.2f}%".format(pct) if pct != 0 else '', startangle=90, colors=colors)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# Adjust autopct to only show non-zero percentages and improve label positioning
for text, autotext in zip(texts, autotexts):
    if autotext.get_text() == '0.0%':  # Hide zero percentages
        text.set_visible(False)
        autotext.set_visible(False)
# Adding a legend
ax.legend(wedges, labels, title="Assets", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.title('Optimal Portfolio Allocation for Maximum Sharpe Ratio')
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
# Filter out assets with zero weights to avoid clutter in the pie chart
filtered_weights = {k: v for k, v in min_vol_cleaned.items() if v > 0}
labels = list(filtered_weights.keys())
sizes = [filtered_weights[asset] for asset in labels]
colors = [asset_colors.get(asset, 'gray') for asset in labels]  # Assign colors
# Creating the pie chart
fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct=lambda pct: "{:.2f}%".format(pct) if pct != 0 else '', startangle=90, colors=colors)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# Adjust autopct to only show non-zero percentages and improve label positioning
for text, autotext in zip(texts, autotexts):
    if autotext.get_text() == '0.0%':  # Hide zero percentages
        text.set_visible(False)
        autotext.set_visible(False)
# Adding a legend
ax.legend(wedges, labels, title="Assets", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1)) 
plt.title('Optimal Portfolio Allocation for Minimum Volatility')
st.pyplot(fig)

# Multi-Objective Genetic Algorithm for Portfolio Optimization
st.header('Optimized Portfolio using Multi-Objective Genetic Algorithm')
def setup_toolbox():
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(assets))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return toolbox
def get_fitness_values(ind):
    return ind.fitness.values
@st.cache_data
def run_genetic_algorithm():
    # Evaluation function for the portfolio
    def eval_portfolio(individual):
        individual = np.array(individual)
        individual /= np.sum(individual)  # Normalize weights
        expected_return = np.dot(individual, mu)
        volatility = np.sqrt(np.dot(individual.T, np.dot(S, individual)))
        sharpe_ratio = expected_return / volatility if volatility != 0 else 0  # Handle division by zero
        return expected_return, volatility, sharpe_ratio
    toolbox.register("evaluate", eval_portfolio)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=20.0, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)
    pop = toolbox.population(n=300)
    hof = tools.ParetoFront()
    stats = tools.Statistics(get_fitness_values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    algorithms.eaMuPlusLambda(pop, toolbox, mu=300, lambda_=300, cxpb=0.7, mutpb=0.3, ngen=40, stats=stats, halloffame=hof, verbose=True)
    return pop, stats, hof

toolbox = setup_toolbox()
# Run the optimization
population, stats, pareto_front = run_genetic_algorithm()

# Plotting the Pareto Front
fig, ax = plt.subplots()
returns = [ind.fitness.values[0] for ind in pareto_front]
volatilities = [ind.fitness.values[1] for ind in pareto_front]
sharpes = [ind.fitness.values[2] for ind in pareto_front]
sc = ax.scatter(returns, volatilities, c=sharpes, cmap='viridis')
plt.colorbar(sc, label='Sharpe Ratio')
ax.set_xlabel('Expected Return')
ax.set_ylabel('Expected Volatility')
ax.set_title('Pareto Front')
st.pyplot(fig)

# Generate dropdown options with a function to ensure they're updated on selection
def get_dropdown_options():
    return {f"Portfolio {i}: Return {ind.fitness.values[0]:.4f}, Risk {ind.fitness.values[1]:.5f}, Sharpe {ind.fitness.values[2]:.3f}": ind 
            for i, ind in enumerate(pareto_front)}

options = get_dropdown_options()

# Use session state to remember the user's choice even after rerunning the script
if 'selected_option' not in st.session_state:
    st.session_state['selected_option'] = list(options.keys())[0]

selected_option = st.selectbox("Select a portfolio to view weights", list(options.keys()), 
                               index=list(options.keys()).index(st.session_state['selected_option']))
st.session_state['selected_option'] = selected_option

# Get the individual based on the selected option and normalize the weights
selected_individual = options[selected_option]
normalized_weights = np.array(selected_individual) / np.sum(selected_individual)

# Display selected portfolio weights
st.header("Selected Portfolio Weights")
st.write(f"For {selected_option}")
for asset, weight in zip(assets, normalized_weights):
    st.write(f"{asset}: {weight:.6f}")


# Sidebar for user inputs
investment = st.sidebar.number_input("Enter investment amount($)", 10, 1000000000, 1000)
objective = st.sidebar.radio(
    "Select your investment objective",
    ["Maximize Sharpe Ratio", "Minimize Volatility", "Maximize Return for Given Risk", "Multi-Objective Genetic Optimization"]
)
max_risk = st.sidebar.slider("Maximum Risk Level (%)", 16, 50, 25) / 100.0 if objective == "Maximize Return for Given Risk" else None

# Function to get weights based on objective
def get_weights(objective, max_risk=None, pareto_front=None):
    if objective == "Maximize Sharpe Ratio":
        ef = EfficientFrontier(mu, S)
        return ef.max_sharpe()
    elif objective == "Minimize Volatility":
        ef = EfficientFrontier(mu, S)
        return ef.min_volatility()
    elif objective == "Maximize Return for Given Risk":
        ef = EfficientFrontier(mu, S)
        ef.efficient_risk(target_volatility=max_risk)
        return ef.clean_weights()
    elif objective == "Multi-Objective Genetic Optimization":
        if pareto_front:
            # Let the user choose from the Pareto front via dropdown
            options = {f"Portfolio {i}: Return {ind.fitness.values[0]:.4f}, Risk {ind.fitness.values[1]:.5f}, Sharpe {ind.fitness.values[2]:.3f}": ind for i, ind in enumerate(pareto_front)}
            chosen = st.sidebar.selectbox("Choose a Portfolio from Pareto Front", list(options.keys()))
            return dict(zip(assets, options[chosen]))
        return None

# Get the latest prices
latest_prices = get_latest_prices(df)

# Display and handle allocations
if objective != "Multi-Objective Genetic Optimization" or pareto_front:
    optimal_weights = get_weights(objective, max_risk, pareto_front if objective == "Multi-Objective Genetic Optimization" else None)
    if optimal_weights:
        da = DiscreteAllocation(optimal_weights, latest_prices, total_portfolio_value=investment)
        allocation, leftover = da.lp_portfolio()
        st.header('Discrete Allocation Based on Your Objective')
        st.write(f"Objective: {objective}")
        st.write(f"Allocation: {allocation}")
        st.write(f"Funds remaining: ${leftover:.2f}")
