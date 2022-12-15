import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics

forecast_reader = csv.reader(open("flu_fct_processed.csv", "r"), delimiter=',')

dates = None
methods = []
forecasts = []
gtruth = []

for (i, row) in enumerate(forecast_reader):
    if i == 0:
        dates = row[1:]
    else:
        if row[0] == "gtruth":
            gtruth = list(map(float, row[1:]))
        else:
            methods.append(row[0])
            forecasts.append(list(map(float, row[1:])))

K = len(methods)
T = len(dates)

ensemble_parameters_reader = csv.reader(open("ensemble_parameters.csv", "r"), delimiter=",")

sigma = {}
w = {}

for (i, row) in enumerate(ensemble_parameters_reader):
    if i == 0:
        continue
    iters = int(row[0])
    sigma_i = float(row[1])
    w_i = list(map(float, row[2:]))
    sigma[iters] = sigma_i
    w[iters] = w_i

regression_parameters_reader = csv.reader(open("regression_parameters.csv", "r"), delimiter=",")

a = []
b = []

for (i, row) in enumerate(regression_parameters_reader):
    if i == 1:
        a = list(map(float, row[1:]))
    elif i == 2:
        b = list(map(float, row[1:]))

x_axis = np.arange(2000, 18000, 50)

iters = 10

f = 0
for k in range(K):
    print(a[k] + b[k] * forecasts[k][-1])
    f += w[iters][k] * norm.pdf(x_axis, forecasts[k][-1], sigma[iters])

plt.plot(x_axis, f)
plt.show()
#
# print(list(forecasts[k][-1] for k in range(K)))
# print(w[iters])
