import csv

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

gtruth_reader = csv.reader(open("flu_hosp_weekly_filt_case_data.csv", "r"), delimiter=",")
rows = list(gtruth_reader)
dates = rows[0][1:]
gtruth_values = list(float(u) for u in rows[-1][1:])

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

import numpy as np

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

from scipy import stats
def get_forecast(method, date):
    for forecast in forecasts:
        if forecast["fct_date"] == date and forecast["method"] == method:
            return forecast["fct_mean"]
def f(k, t):
    return get_forecast(methods[k], dates[start_index + t])
def y(t):
    return gtruth_values[start_index + t]
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

for _ in range(100):
    EM_iteration()

print(z)
print(w)
print(sigma)

import matplotlib.pyplot as plt

plt_dates = dates[start_index:]
plt.plot(plt_dates, get_gtruth_until(plt_dates[-1]))
for method in methods:
    plt.plot(plt_dates, get_forecasts_until(plt_dates[-1], method))
plt.legend(labels=["gtruth"] + methods)
plt.xticks(rotation=30)
plt.gcf().subplots_adjust(bottom=0.2)
plt.show()