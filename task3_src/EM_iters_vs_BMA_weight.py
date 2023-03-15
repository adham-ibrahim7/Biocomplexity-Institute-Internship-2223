import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.ensemble_forecast import EnsembleForecast

# this code causes an error because the EM_iters variable needs to be passed to the class
# not worth fixing right now

if __name__ == "__main__":
    in_filename = "../task2_data/top_10_pop_all_step_ahead.csv"
    # out_filename = "../task2_data/experiment_1.csv"

    all_data = pd.read_csv(in_filename, dtype={"cnty": "str"})

    all_counties = all_data.cnty.unique()
    all_methods = all_data.method.unique()
    all_dates = all_data.horizon.unique()
    all_step_aheads = all_data.step_ahead.unique()

    counties = list(all_counties)
    # this county has less data available
    counties.remove('12086')
    # for the first 45 weeks, these methods all have data
    training_methods = ['AR', 'ARIMA', 'AR_spatial', 'ENKF', 'lstm']

    lead_time = 8
    step_ahead = '1-step_ahead'
    county = counties[0]

    for EM_iters in [10, 20, 50, 100]:
        pointsX = []
        pointsY = []

        for horizon_index in range(20, 30):
            horizon = all_dates[horizon_index]
            training_dates = all_dates[horizon_index - lead_time: horizon_index]

            ensemble = EnsembleForecast(all_data, county, training_methods, training_dates, horizon, step_ahead)

            # ensemble.plot_forecasts()

            shapley = ensemble.approximate_shapley(payoff=ensemble.squared_error_payoff)

            print("horizon:", horizon)
            print("Shapley value:", shapley)
            print("BMA weights:", ensemble.weights)

            pointsX += list(np.array(shapley) / np.sum(shapley))
            pointsY += list(ensemble.weights)

            print()

        plt.clf()
        plt.scatter(pointsX, pointsY)
        plt.xlabel("Shapley Value")
        plt.ylabel("Weight Assigned by BMA")
        plt.title("EM iters={}".format(EM_iters))
        plt.show()