import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad as integral

forecast_reader = csv.reader(open("flu_fct_processed.csv", "r"), delimiter=',')

dates = None
methods = []
raw_forecasts = []
gtruth = []

for (i, row) in enumerate(forecast_reader):
    if i == 0:
        dates = row[1:]
    else:
        if row[0] == "gtruth":
            gtruth = list(map(float, row[1:]))
        else:
            methods.append(row[0])
            raw_forecasts.append(list(map(float, row[1:])))

K = len(methods)
T = len(dates)

ensemble_parameters_reader = csv.reader(open("ensemble_parameters.csv", "r"), delimiter=",")

sigma_from_iters = {}
w_from_iters = {}

for (i, row) in enumerate(ensemble_parameters_reader):
    if i == 0:
        continue
    EM_iters = int(row[0])
    sigma_i = float(row[1])
    w_i = list(map(float, row[2:]))
    sigma_from_iters[EM_iters] = sigma_i
    w_from_iters[EM_iters] = w_i

regression_parameters_reader = csv.reader(open("regression_parameters.csv", "r"), delimiter=",")

a = []
b = []

for (i, row) in enumerate(regression_parameters_reader):
    if i == 1:
        a = list(map(float, row[1:]))
    elif i == 2:
        b = list(map(float, row[1:]))

################################################################################################
# PLOT THE BMA PDF
################################################################################################

def plot_pdf(sigma, w, x_axis):
    def ensemble_pdf(curr_sigma, X):
        pdf = 0
        for k in range(K):
            # print(a[k] + b[k] * forecasts[k][-1])
            pdf += w[k] * norm.pdf(X, a[k] + b[k] * raw_forecasts[k][-1], curr_sigma)
        return pdf

    pdf = lambda x: ensemble_pdf(sigma, x)

    plt.plot(x_axis, ensemble_pdf(sigma, x_axis))

    for u in [.5, .95]:
        l, r = get_CI(u, sigma)
        plt.axvline(x=l, color="black", linestyle="dashed")
        plt.axvline(x=r, color="black", linestyle="dashed")
    plt.axvline(x=gtruth[-1], color="black")

    plt.show()

EM_iters = 10
sigma = sigma_from_iters[EM_iters]
w = w_from_iters[EM_iters]

# plot_pdf(sigma, w)

################################################################################################
# GENERATE SAMPLES AND COMPUTE 50%, 95% CI
################################################################################################

def sample(sigma):
    k = np.random.choice(list(range(K)), p=w)
    mean = a[k] + b[k] * raw_forecasts[k][-1]
    return np.random.normal(mean, sigma)

def get_CI(CI_size, sigma):
    n_samples = 1000
    samples = np.empty(n_samples)
    for i in range(n_samples):
        samples[i] = sample(sigma)

    samples = sorted(samples)

    start_index = int(n_samples * (.5 * (1 - CI_size)))
    end_index = n_samples - start_index
    return samples[start_index], samples[end_index]

print("BEFORE CALIBRATION: sigma=", sigma)
print("MIDDLE 50%:", get_CI(.5, sigma))
print("MIDDLE 95%:", get_CI(.95, sigma))
plot_pdf(sigma, w, np.arange(6000, 16000, 10))

################################################################################################
# CALIBRATION OF SIGMA TO MAXIMIZE CRPS SCORE
################################################################################################

def ensemble_cdf(curr_sigma, X):
    cdf = 0
    for k in range(K):
        cdf += w[k] * norm.cdf(X, a[k] + b[k] * raw_forecasts[k][-1], curr_sigma)
    return cdf

def CRPS(sigma, x):
    cdf = lambda t: ensemble_cdf(sigma, t)

    def f_1(t):
        y = cdf(t)
        return y * y

    def f_2(t):
        y = cdf(t)
        return (1 - y) * (1 - y)

    return integral(f_1, 0, x)[0] + integral(f_2, x, 30000)[0]

# print(sigma)
# print(CRPS(sigma, gtruth[-1]))

x_axis = np.arange(0, 400, 20)

lo = 0
hi = 400
count = 0
while hi - lo > 1:
    count += 1

    mid_left = lo + (hi - lo) / 3
    mid_right = lo + (hi - lo) * 2 / 3
    if CRPS(mid_left, gtruth[-1]) < CRPS(mid_right, gtruth[-1]):
        hi = mid_right
    else:
        lo = mid_left

calibrated_sigma = lo

print("AFTER CALIBRATION: calibrated_sigma=", calibrated_sigma)
print("MIDDLE 50%:", get_CI(.5, calibrated_sigma))
print("MIDDLE 95%:", get_CI(.95, calibrated_sigma))
plot_pdf(calibrated_sigma, w, np.arange(10000, 12500, 10))

print("GROUND TRUTH on " + dates[-1] + ":")
print(gtruth[-1])

#
# plt.plot(x_axis, list(map(lambda s: CRPS(s, gtruth[-1]), x_axis)))
# plt.show()