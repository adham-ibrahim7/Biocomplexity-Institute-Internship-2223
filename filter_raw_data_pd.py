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

data_frame = pd.read_csv(in_filename, dtype={"cnty": "str"})
print("Total rows:", len(data_frame))

filtered = data_frame.query(filter_str)
print(filtered)

with open(out_filename, "w", newline='') as out_file:
    filtered.to_csv(path_or_buf=out_file, sep=",", index=False)