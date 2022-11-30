import csv

csv_reader = csv.reader(open("flu_fct_files_2022-11-13.csv", "r"), delimiter=',')

forecasts = []

line_count = 0
for row in csv_reader:
    if line_count > 0:
        forecast = {
            "fct_date": row[0],
            "fct_mean": float(row[1]),
            "cnty": row[2],
            "horizon": row[3],
            "method": row[4],
            "step_ahead": row[5]
        }
        forecasts.append(forecast)
    line_count += 1

US_forecasts = list(filter(lambda forecast: forecast["cnty"] == "US" and forecast["step_ahead"] == "1-step_ahead", forecasts))
US_forecasts = sorted(US_forecasts, key=lambda forecast: forecast["horizon"])

out = open("US_1-step_ahead-flu_fct_files_2022-11-13.csv", "w", newline='')
csv_writer = csv.writer(out)
csv_writer.writerow("fct_date,fct_mean,cnty,horizon,method,step_ahead".split(","))
for forecast in US_forecasts:
    csv_writer.writerow(str(u) for u in forecast.values())
out.close()