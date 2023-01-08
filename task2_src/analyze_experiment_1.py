import numpy as np
import pandas as pd

class Result:
    def __init__(self, experiment_df, county, all_lead_times, all_step_aheads):
        self.ensemble_means = {}

        cnty_df = experiment_df.query("cnty == '{}'".format(county))
        for lead_time in all_lead_times:
            self.ensemble_means[lead_time] = {}
            for step_ahead in all_step_aheads:
                target = cnty_df.query("lead_time == {} and step_ahead == '{}'".format(lead_time, step_ahead))
                mean = target.at[target.index[0], 'ensemble_mean']
                self.ensemble_means[lead_time][step_ahead] = mean

if __name__ == "__main__":
    in_filename = "../task2_data/experiment_1.csv"
    all_data_df = pd.read_csv(in_filename, dtype={"cnty": "str"})

    all_counties = all_data_df.cnty.unique()
    all_lead_times = all_data_df.lead_time.unique()
    all_step_aheads = all_data_df.step_ahead.unique()

    # def get(query_str, field):
    #     target = all_data_df.query(query_str)
    #     return target.at[target.index[0], field]

    c = 10.0
    for lead_time in all_lead_times:
        df = all_data_df.query("lead_time == {}".format(lead_time))
        ratio = df.ensemble_mean / df.true
        print(lead_time,
              np.mean(df.calibrated_sigma),
              np.mean(ratio[ratio < c]),
              np.mean(df.captured_50),
              np.mean(df.captured_75),
              np.mean(df.captured_95)
        )
