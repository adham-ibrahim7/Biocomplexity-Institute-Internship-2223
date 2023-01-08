import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from stopwatch import Stopwatch

from task2_src.ensemble_forecast import EnsembleForecast

if __name__ == "__main__":
    forecasts_filename = "../task2_data/top_10_pop_all_step_ahead.csv"
    out_filename = "../task2_data/experiment_2.csv"

    forecasts_df = pd.read_csv(forecasts_filename, dtype={"cnty": "str"})
    try:
        saved_results_df = pd.read_csv(out_filename, dtype={"cnty": "str"}, index_col=False)
    except:
        saved_results_df = None

    all_counties = forecasts_df.cnty.unique()
    all_methods = forecasts_df.method.unique()
    all_dates = forecasts_df.horizon.unique()
    all_step_aheads = forecasts_df.step_ahead.unique()

    counties = list(all_counties)
    # this county has less data available
    counties.remove('12086')
    # for the first 45 weeks, these methods all have data
    training_methods = ['AR', 'ARIMA', 'AR_spatial', 'ENKF']

    all_horizons = []

    frames = []

    stopwatch = Stopwatch()
    for horizon_index in range(15, 45):
        horizon = all_dates[horizon_index]
        all_horizons.append(horizon)

        for lead_time in [8]:
            training_dates = all_dates[horizon_index - lead_time: horizon_index]

            for county in counties[:1]:
                for step_ahead in all_step_aheads:
                    if saved_results_df is not None:
                        saved_result = saved_results_df.query("cnty == '{}' and horizon == '{}' and lead_time == {} and step_ahead == '{}'".format(county, horizon, lead_time, step_ahead))
                        if not saved_result.empty:
                            frames.append(saved_result)
                            continue

                    forecast = EnsembleForecast(forecasts_df, county, training_methods, training_dates, horizon, step_ahead)

                    row = {
                        'horizon': [horizon],
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

    all_results_df = pd.concat(frames, ignore_index=True)
    all_results_df.to_csv(out_filename, sep=',', index=False)

    print(all_results_df)

    for step_ahead in all_step_aheads:
        df = all_results_df.query("step_ahead == '{}'".format(step_ahead))

        plt.clf()
        plt.plot(all_horizons, df.ensemble_mean)
        plt.plot(all_horizons, df.true)
        plt.xticks(all_horizons[::3], rotation=70)
        plt.gcf().subplots_adjust(bottom=0.25)
        plt.title("cnty='{}', lead_time='{}'\nstep_ahead='{}".format(county, lead_time, step_ahead))

        plt.show()

    stopwatch.stop()
    print("TOTAL TIME", stopwatch)