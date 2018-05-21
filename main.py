import matplotlib.pyplot as plt
import constants
from get_data import AllData
from strategy import Strategy
import numpy as np


np.random.seed(1234)

# GET ALL DATA FROM FILE AND CREATE MORE INPUTS
dj = AllData(constants.INPUT_FILE)
dj.add_indicators()
dj.create_target()
dj.clean_data()

# CREATE NEURAL NETWORK AND TRAIN IT
my_strategy = Strategy(dj.data.loc[:, dj.data.columns != 'DATE'].values,
                       dj.data['DATE'].values,
                       dj.df_target,
                       dj.first_idx_ret-1)
dj = None
my_strategy.create_lstm()
my_strategy.train()

# TEST NEURAL NETWORK ON FINANCIAL PORTFOLIO
my_strategy.create_benchmark()
my_strategy.create_portfolio()
my_strategy.test()


# PLOT PORTFOLIO RETURNS
plt.plot(my_strategy.ptf.values, '--', lw=3, label='PTF')
plt.plot(my_strategy.bmk.values, '-', lw=3, label='BMK')
plt.legend()
plt.show()
print('the end')
