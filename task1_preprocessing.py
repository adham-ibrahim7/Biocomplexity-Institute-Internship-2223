import csv
import numpy as np
from scipy import stats

# Reading in forecasts

forecast_reader = csv.reader(open("US_1-step_ahead-flu_fct_files_2022-11-13.csv", "r"), delimiter=',')

forecasts = []
line_count = 0
for row in forecast_reader:
    if line_count > 0:
        forecast = {
            "fct_date": row[0],
            "fct_mean": float(row[1]),
            "cnty": row[2],
            "horizon": row[3],
            "method": row[4],
            "step_ahead": row[5]
        }

        if forecast["fct_date"] < "2022-10-02":
            continue

        forecasts.append(forecast)
    line_count += 1

# Reading in ground truth values

gtruth_reader = csv.reader(open("flu_hosp_weekly_filt_case_data.csv", "r"), delimiter=",")
rows = list(gtruth_reader)
dates = rows[0][1:]
gtruth_values = list(float(u) for u in rows[-1][1:])

# methods for getting forecast, ground truth by date/method
def get_forecasts_until(date, method):
    values = []
    for forecast in forecasts:
        if forecast["fct_date"] > date:
            break
        if forecast["method"] == method:
            values.append(forecast["fct_mean"])
    return values

start_index = dates.index("2022-10-02")
def get_gtruth_until(date):
    i = dates.index(date)
    return gtruth_values[start_index:i+1]

def get_forecast(method, date):
    for forecast in forecasts:
        if forecast["fct_date"] == date and forecast["method"] == method:
            return forecast["fct_mean"]
def f(k, t):
    return get_forecast(methods[k], dates[start_index + t])
def y(t):
    return gtruth_values[start_index + t]

# method to perform linear regressiono of Y on X
def regression(X, Y):
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)

    U = X-X_mean
    V = Y-Y_mean
    b = np.dot(U, V) / np.dot(U, U)
    a = Y_mean - b * X_mean
    return a, b

date = "2022-11-13"
methods = "ARIMA_LO lstm EnKF phase-model ES_LO".split()

# Compute a_k, b_k for each method: the regression parameters of the forecast's history on the ground truth

regression_parameters = []
Y = get_gtruth_until(date)
for i in range(len(methods)):
    X = get_forecasts_until(date, methods[i])
    regression_parameters.append(regression(X, Y))
    print(regression_parameters[i])

K = len(methods)
T = len(Y)

z = np.full((K, T), 1 / K)
w = np.full(K, 1 / K)
sigma = 100

# Expectation-maximization algorithm to find w, sigma

def EM_iteration():
    global w, z, sigma

    q = np.zeros((K, T))
    for k in range(K):
        a, b = regression_parameters[k]
        for t in range(T):
            q[k][t] = w[k] * stats.norm(a + b * f(k, t), sigma).pdf(y(t))
    temp = np.zeros((K, T))
    for k in range(K):
        for t in range(T):
            temp[k][t] = q[k][t] / sum(q[j][t] for j in range(K))
    z = temp

    # print(z)

    w = np.zeros(K)
    for k in range(K):
        w[k] = np.mean(z[k])

    sigma = np.sqrt(1/T * sum(sum(z[k][t] * np.square(f(k, t)-y(t)) for k in range(K)) for t in range(T)))

# Store forecast, ground truth in a nicer format in `flu_fct_processed.csv`

out = open("flu_fct_processed.csv", "w", newline='')
forecast_writer = csv.writer(out)
forecast_writer.writerow(["data"] + list(dates[start_index+t] for t in range(T)))
for k in range(K):
    forecast_writer.writerow([methods[k]] + list(str(f(k, t)) for t in range(T)))
forecast_writer.writerow(["gtruth"] + list(str(y(t)) for t in range(T)))
out.close()

# perform EM and store values to `ensemble_parameters.csv`

iters = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
prev = 0

out = open("ensemble_parameters.csv", "w", newline='')
parameters_writer = csv.writer(out)
parameters_writer.writerow(["EM_iters", "sigma"] + methods)
for curr in iters:
    for _ in range(curr - prev):
        EM_iteration()
    prev = curr
    parameters_writer.writerow([str(curr), str(sigma)] + list(map(str, w)))
    print(w)
    print(sigma)
out.close()

# Store regression parameters

out = open("regression_parameters.csv", "w", newline='')
parameters_writer = csv.writer(out)
parameters_writer.writerow(["data"] + methods)
parameters_writer.writerow(["a"] + list(str(regression_parameters[k][0]) for k in range(K)))
parameters_writer.writerow(["b"] + list(str(regression_parameters[k][1]) for k in range(K)))
out.close()

# import matplotlib.pyplot as plt
#
# plt_dates = dates[start_index:]
# plt.plot(plt_dates, get_gtruth_until(plt_dates[-1]))
# for method in methods:
#     plt.plot(plt_dates, get_forecasts_until(plt_dates[-1], method))
# plt.legend(labels=["gtruth"] + methods)
# plt.xticks(rotation=30)
# plt.gcf().subplots_adjust(bottom=0.2)
# plt.show()