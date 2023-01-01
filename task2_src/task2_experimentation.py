import numpy as np
import pandas as pd
from stopwatch import Stopwatch

from task2_src.task2_main import EnsembleForecast

if __name__ == "__main__":
    in_filename = "../task2_data/top_10_pop_all_step_ahead.csv"

    all_data = pd.read_csv(in_filename, dtype={"cnty": "str"})

    all_counties = all_data.cnty.unique()
    all_methods = all_data.method.unique()
    all_dates = all_data.horizon.unique()
    all_step_aheads = all_data.step_ahead.unique()

    training_methods = ['AR', 'ARIMA', 'AR_spatial', 'ENKF', 'PatchSim_adpt']
    # training_methods = ['PatchSim_adpt', 'AR_spatial']
    horizon_index = 15
    T = 8
    horizon = all_dates[horizon_index]
    training_dates = all_dates[horizon_index-T:horizon_index]
    print(horizon)

    stopwatch = Stopwatch()

    for county in all_counties[:5]:
        for step_ahead in all_step_aheads:
            print("-------cnty={}, step_ahead={}----------".format(county, step_ahead))

            forecast = EnsembleForecast(all_data, county, training_methods, training_dates, horizon, step_ahead)

            # print("weights:", forecast.weights)
            # print("uncalibrated sigma:", forecast.uncalibrated_sigma)
            # print("calibrated sigma:", forecast.calibrated_sigma)

            filename_info = "{}-weeks-{}-{}-{}".format(T, county, horizon, step_ahead)
            forecast.plot_pdf(forecast.horizon, forecast.calibrated_sigma,
                              show=False, save_to="../task2_figures/{}-pdf.png".format(filename_info))
            forecast.plot_forecasts(forecast.training_methods, forecast.all_dates,
                                    show=False, save_to="../task2_figures/{}-forecasts.png".format(filename_info))

            for u in [0.5, 0.75, 0.95]:
                left, right = forecast.get_confidence_interval(forecast.horizon, forecast.calibrated_sigma,
                                                               interval_size=u)
                print(("CAPTURED" if left < forecast.ground_truth[forecast.horizon] < right else "NOT CAPTURED") +
                      " BY {}-CI".format(u))

    stopwatch.stop()
    print("TOTAL TIME", stopwatch)