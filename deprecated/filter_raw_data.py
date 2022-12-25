import csv

in_filename = "../raw_data/covid_cnty_cases_all_methods.csv"
out_filename = "task2_data/temp.csv"
counties = ["01001", "44005"]
step_ahead = ["1-step_ahead", "2-step_ahead"]

def keep_forecast(forecast):
    return forecast["cnty"] in counties and forecast["step_ahead"] in step_ahead

csv_reader = csv.reader(open(in_filename, "r"), delimiter=',')
# num_forecasts = sum(1 for _ in csv_reader)
# print("Total forecasts:", num_forecasts)

forecasts = []

header = None
for row in csv_reader:
    if csv_reader.line_num == 1:
        header = row
    else:
        forecast = {}
        for (i, entry) in enumerate(header):
            forecast[entry] = row[i]

        if keep_forecast(forecast):
            forecasts.append(forecast)

# desired_forecasts = list(filter(keep_forecast, forecasts))
# desired_forecasts = sorted(desired_forecasts, key=lambda forecast: forecast["horizon"])

print("Done reading file. Num saved forecasts:", len(forecasts))

out = open(out_filename, "w", newline='')
csv_writer = csv.writer(out)
csv_writer.writerow(header)
for forecast in forecasts:
    csv_writer.writerow(str(u) for u in forecast.values())
out.close()