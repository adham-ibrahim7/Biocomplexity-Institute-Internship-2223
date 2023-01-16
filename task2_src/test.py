import numpy as np
import pandas as pd
from stopwatch import Stopwatch

from task2_src.ensemble_forecast import EnsembleForecast

if __name__ == "__main__":
    in_filename = "../task2_data/top_10_pop_all_step_ahead.csv"
    out_filename = "../task2_data/experiment_1.csv"

    all_data = pd.read_csv(in_filename, dtype={"cnty": "str"})

    all_counties = all_data.cnty.unique()
    all_methods = all_data.method.unique()
    all_dates = all_data.horizon.unique()
    all_step_aheads = all_data.step_ahead.unique()

    counties = list(all_counties)
    # this county has less data available
    counties.remove('12086')
    # for the first 45 weeks, these methods all have data
    training_methods = ['AR', 'ARIMA', 'AR_spatial', 'ENKF']

    horizon_index = 44
    lead_time = 8

    county = counties[0]
    horizon = all_dates[horizon_index]
    training_dates = all_dates[horizon_index-lead_time: horizon_index]
    step_ahead = '1-step_ahead'

    forecast = EnsembleForecast(all_data, county, training_methods, training_dates, horizon, step_ahead, print_status=True)
    print(forecast.uncalibrated_sigma)
    print(forecast.calibrated_sigma)
