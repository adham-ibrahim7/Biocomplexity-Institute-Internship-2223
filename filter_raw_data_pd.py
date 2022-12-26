import pandas as pd

in_filename = "raw_data/covid_cnty_cases_all_methods.csv"
out_filename = "task2_data/top_10_pop_all_step_ahead.csv"

config_filename = "task2_data/top_10_pop_fips.txt"
counties = open(config_filename, "r").read().split(",")
step_ahead = ["1-step_ahead", "2-step_ahead", "3-step_ahead", "4-step_ahead"]

filter_str = "cnty in @counties and step_ahead in @step_ahead"

########################################################################

print("counties:", counties)
print("step ahead:", step_ahead)

raw_df = pd.read_csv(in_filename, dtype={"cnty": "str"})
print("Total rows:", len(raw_df))

final_df = raw_df.query(filter_str)\
    .drop_duplicates()\
    .drop('fct_std', axis=1)\
    .sort_values(by=['cnty', 'method', 'horizon', 'step_ahead'])

print(final_df)

with open(out_filename, "w", newline='') as out_file:
    final_df.to_csv(path_or_buf=out_file, sep=",", index=False)