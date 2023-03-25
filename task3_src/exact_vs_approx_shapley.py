import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from stopwatch import Stopwatch

from src.Shapley import Shapley
from src.ensemble_forecast import EnsembleForecast

if __name__ == "__main__":
    forecasts_filename = "../task2_data/top_10_pop_all_step_ahead.csv"
    forecasts_df = pd.read_csv(forecasts_filename, dtype={"cnty": "str"})

    all_counties = forecasts_df.cnty.unique()
    all_methods = forecasts_df.method.unique()
    all_dates = forecasts_df.horizon.unique()
    all_step_aheads = forecasts_df.step_ahead.unique()

    counties = list(all_counties)
    # this county has less data available
    counties.remove('12086')
    # for the first 45 weeks, these methods all have data
    training_methods = ['AR', 'ARIMA', 'AR_spatial', 'ENKF', 'lstm']

    all_horizons = []

    print(all_dates[73])

    for horizon_index in range(30, 31):
        horizon = all_dates[horizon_index]
        all_horizons.append(horizon)

        for lead_time in [8]:
            training_dates = all_dates[horizon_index - lead_time: horizon_index]

            for county in counties:
                for step_ahead in all_step_aheads[:1]:
                    print(county, all_dates[horizon_index])

                    shapley = Shapley(forecasts_df, county, training_methods, training_dates, horizon, step_ahead)

                    results = []

                    for _ in range(100):
                        results.append(shapley.approximate_shapley(num_permutations=20))

                    print("means:  ", np.mean(results, axis=0))
                    print("stddevs:", np.std(results, axis=0))
                    print("exact:  ", shapley.exact_shapley())

                    forecast = EnsembleForecast(forecasts_df, county, training_methods, training_dates, horizon, step_ahead)
                    print("BMA weights: ", forecast.weights)