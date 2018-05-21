import matplotlib.pyplot as plt
import constants
from get_data import AllData
from strategy import Strategy
import numpy as np


np.random.seed(1234)

dj = AllData(constants.INPUT_FILE)
dj.add_indicators()
dj.create_target()
dj.clean_data()

my_strategy = Strategy(dj.data.loc[:, dj.data.columns != 'DATE'].values,
                       dj.data['DATE'].values,
                       dj.df_target,
                       dj.first_idx_ret-1)
dj = None
my_strategy.create_lstm()
my_strategy.train()

my_strategy.create_benchmark()
my_strategy.create_portfolio()
my_strategy.test()


plt.plot(my_strategy.ptf.values, '--', lw=3, label='PTF')
plt.plot(my_strategy.bmk.values, '-', lw=3, label='BMK')
plt.legend()
plt.show()
print('the end')
