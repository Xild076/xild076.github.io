import StockData
import Policy

# List of stocks - You can change this and add more
sdata = ['AAPL']
# List of FRED data - You can change this and add more
fdata = ['T10YIE', 'T10Y2Y', 'MORTGAGE30US', 'SNDR', 'MMTY', 'DGS10', 'TB3MS', 
                                'FEDFUNDS', 'SOFR', 'BAMLH0A0HYM2', 'DTWEXBGS', 'BOGMBASE', 'WRESBAL', 
                                'RCCCBBALTOT', 'PCE', 'GDP', 'CPIAUCSL', 'INDPRO', 'REAINTRATREARAT10Y', 
                                'RSXFS', 'COMPOUT']


# Count of how many days will be inputted
day_count = 20
# Number of repetitions through
scale = 2
# Number of one directional percentage choices
act_range = 8
# Max percentage of choice variation
percent_max = 0.15

fin_env = StockData.FinanceEnv(sdata, fdata, day_count, scale, act_range, percent_max)

# Learning rate
l_rate = 1
# Discount rate
discount_rate = 0.99
# Decay rate
decay_rate = 0.99
# Exploration rate
e_rate = 0.1
# Hidden layers
h_layers = 100
p_alg = Policy.PolicyAlgorithm(fin_env, l_rate, discount_rate, decay_rate, e_rate, h_layers)

# Training amount
epochs = 2000
# Batch size
b_size = 2
# Max steps
m_step = 100

p_alg.train(2000, 2, m_step)

#Testing
for i in range(3):
    p_alg.test(100)

