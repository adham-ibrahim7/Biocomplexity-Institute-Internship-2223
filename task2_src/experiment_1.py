import numpy as np
import pandas as pd
from stopwatch import Stopwatch

from src.ensemble_forecast import EnsembleForecast

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
    horizon = all_dates[horizon_index]

    frames = []

    stopwatch = Stopwatch()
    for lead_time in range(4, 12, 2):
        training_dates = all_dates[horizon_index - lead_time : horizon_index]

        for county in counties:
            for step_ahead in all_step_aheads:
                forecast = EnsembleForecast(all_data, county, training_methods, training_dates, horizon, step_ahead)

                row = {
                    'lead_time': [lead_time],
                    'cnty': [county],
                    'step_ahead': [step_ahead],
                    'ensemble_mean': [forecast.get_mean(forecast.horizon)],
                    'true': [forecast.ground_truth[forecast.horizon]],
                    'calibrated_sigma': [forecast.calibrated_sigma]
                }

                for interval_size, label in [(0.5, '50'), (0.75, '75'), (0.95, '95')]:
                    left, right = forecast.get_confidence_interval(forecast.horizon, forecast.calibrated_sigma,
                                                                   interval_size=interval_size)
                    captured = left < forecast.ground_truth[forecast.horizon] < right

                    row['captured_{}'.format(label)] = ['1'] if captured else ['0']

                frames.append(pd.DataFrame(row))

                print(row)

    out_df = pd.concat(frames, ignore_index=True)
    out_df.to_csv(out_filename, sep=',')

    stopwatch.stop()
    print("TOTAL TIME", stopwatch)
