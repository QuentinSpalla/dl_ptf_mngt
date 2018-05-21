import numpy as np


class Portfolio:
    """
    Contains financial portfolio features
    """
    def __init__(self, nbr_assets, transac_fee_rate, initial_value, name=''):
        """
        :param nbr_assets: long, number of potential assets in the portfolio
        :param transac_fee_rate: float
        :param initial_value: float, initial value of the portfolio
        :param name: string
        """
        self.weights = np.zeros([nbr_assets, 1])
        self.last_weights = np.zeros([nbr_assets, 1])
        self.curt_time = 0
        self.curt_return = 0
        self.curt_value = initial_value
        self.curt_transac_value = 0
        self.values = [initial_value]
        self.transac_values = [0]
        self.name = name
        self.transac_fee_rate = transac_fee_rate

    def update_weights(self, new_w):
        """
        Updates portfolio's weights
        :param new_w: ndarray, new weights
        """
        self.last_weights = self.weights
        self.weights = new_w

    def update_time(self):
        self.curt_time += 1

    def update_weights_inv_val(self, values):
        """
        Updates portfolio's weights proportional to values' inverse
        :param values: ndarray
        """
        temp_w = values / sum(values)
        self.update_weights(temp_w)

    def update_weights_inv_rdt(self, rdt):
        """
        Updates portfolio's weights proportional to values' inverse
        :param rdt: ndarray, financial returns of all assets
        """
        temp_w_r = self.weights * rdt
        self.update_weights(temp_w_r / sum(temp_w_r))

    def update_val_list(self):
        """
        Adds portfolio's value and transaction's value to lists (get historical path)
        """
        self.values.append(self.curt_value)
        self.transac_values.append(self.curt_transac_value)

    def compute_transaction_fees(self):
        self.curt_transac_value = float(sum(abs(self.weights - self.last_weights))
                                        * self.curt_value
                                        * self.transac_fee_rate)

    def compute_return(self, assets_ret):
        """
        Computes portfolio's financial return
        :param assets_ret: ndarray, financial returns of all assets
        """
        self.curt_return = float(np.dot(self.weights.T, assets_ret))

    def compute_value(self):
        """
        Computes portfolio's financial value
        """
        self.curt_value = float(self.curt_value * (1.+self.curt_return))
