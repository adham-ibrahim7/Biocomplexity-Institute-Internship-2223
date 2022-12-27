import pandas as pd
import timeit
import numpy as np
from scipy import stats

in_filename = "../task2_data/top_10_pop_all_step_ahead.csv"

all_data = pd.read_csv(in_filename, dtype={"cnty": "str"})

print("Processing raw data.")

training_counties = all_data.cnty.unique()[:5]
all_dates = all_data.horizon.unique()
training_methods = all_data.method.unique()

forecasts: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
ground_truth: dict[str, dict[str, float]] = {}
for county in training_counties:
    county_df = all_data.query("cnty == @county")
    forecasts[county] = {}
    ground_truth[county] = {}
    for method in training_methods:
        county_method_df = county_df.query("method == @method")
        forecasts[county][method] = {}
        for date in all_dates:
            forecasts[county][method][date] = {}
            county_method_date_df = county_method_df.query("horizon == @date")

            if not county_method_date_df.empty:
                for i in range(len(county_method_date_df)):
                    step_ahead = county_method_date_df.at[county_method_date_df.index[i], 'step_ahead']
                    fct_mean = county_method_date_df.at[county_method_date_df.index[i], 'fct_mean']
                    forecasts[county][method][date][step_ahead] = fct_mean

            if date not in ground_truth[county]:
                temp = county_method_df.query("fct_date == @date")
                if temp.empty:
                    continue
                ground_truth[county][date] = temp.at[temp.index[0], 'true']

print("Done processing raw data.")

################################################################################################

# date_index = 20
# horizon = all_dates[date_index]
step_ahead = '1-step_ahead'
# fct_date = all_dates[date_index+1]
# county = '04013'
training_methods = ['AR', 'ARIMA', 'AR_spatial', 'ENKF', 'PatchSim_adpt']

K = len(training_methods)
S = len(training_counties)
T = 15

training_dates = all_dates[1:T+1]

################################################################################################
# PLOTTING FOR DEBUGGING
################################################################################################

import matplotlib.pyplot as plt

# Returns days filtered from dates when the method had a forecast output
def filter_valid_dates(method, dates):
    return np.array(list(filter(lambda t: step_ahead in forecasts[county][method][t], dates)))

def plot(plt_counties, plt_methods, plt_dates):
    for county in plt_counties:
        plt.clf()
        plt.plot(plt_dates, list(map(lambda x: ground_truth[county][x], plt_dates)), color='black')
        for method in plt_methods:
            valid_dates = filter_valid_dates(method, plt_dates)
            plt.plot(valid_dates, list(map(lambda x: forecasts[county][method][x][step_ahead], valid_dates)))
        # plt.xticks(list(all_dates[i] for i in range(0, len(all_dates), 10)), rotation=30)
        plt.legend(labels=['gtruth'] + plt_methods)
        plt.title(county + " forecasts")
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.show()

def plot_error(plt_counties, plt_methods, plt_dates):
    for county in plt_counties:
        plt.clf()
        for method in plt_methods:
            valid_dates = filter_valid_dates(method, plt_dates)
            plt.plot(valid_dates, list(map(lambda x: forecasts[county][method][x][step_ahead] / ground_truth[county][x], valid_dates)))
        # plt.xticks(list(all_dates[i] for i in range(0, len(all_dates), 10)), rotation=30)
        plt.legend(labels=plt_methods)
        plt.title(county + " error")
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.show()

plt_counties = training_counties[:3]
plt_methods = ['AR_spatial']
plt_dates = all_dates[1:]

plot(plt_counties, plt_methods, plt_dates)
plot_error(plt_counties, plt_methods, plt_dates)

################################################################################################
# LINEAR REGRESSION TO GET PARAMETERS a[k], b[k] FOR EACH METHOD
################################################################################################

# method to perform linear regression of Y on X
def regression(X, Y):
    if len(X) == 0:
        return -1, -1

    X_mean = np.mean(X)
    Y_mean = np.mean(Y)

    U = X-X_mean
    V = Y-Y_mean
    b = np.dot(U, V) / np.dot(U, U)
    a = Y_mean - b * X_mean
    return a, b

# Compute a_k, b_k for each method: the regression parameters of the forecast's history on the ground truth

# valid_dates = {}
regression_parameters = {}

for method in training_methods:
    X = []
    Y = []
    for county in training_counties:

        # print(method.ljust(20), end=' ')
        # for day in training_dates:
        #     print('T' if day in valid_dates[method] else ' ', end='')
        # print()

        X += list(map(lambda x: forecasts[county][method][x][step_ahead], training_dates))
        Y += list(map(lambda x: ground_truth[county][x], training_dates))

    regression_parameters[method] = regression(X, Y)
    print(method, regression_parameters[method])

################################################################################################
# EXPECTATION-MAXIMIZATION ALGORITHM TO FIND w, sigma
################################################################################################

def EM_iteration(f, y, K, T, w, sigma):
    # global w, z, sigma

    q = np.zeros((S, K, T))
    for s in range(S):
        for k in range(K):
            a, b = regression_parameters[training_methods[k]]
            for t in range(T):
                q[s][k][t] = w[k] * stats.norm(a + b * f(s, k, t), sigma).pdf(y(s, t))
    z = np.zeros((S, K, T))
    for s in range(S):
        for k in range(K):
            for t in range(T):
                z[s][k][t] = q[s][k][t] / sum(q[s][j][t] for j in range(K))

    w = np.zeros(K)
    for k in range(K):
        w[k] = np.mean(z[:, k, :])

    variance = 0
    for s in range(S):
        for t in range(T):
            for k in range(K):
                variance += z[s][k][t] * np.square(y(s, t) - f(s, k, t))
    variance /= (S * T)

    sigma = np.sqrt(variance)

    return w, sigma

def f(s, k, t):
    return forecasts[training_counties[s]][training_methods[k]][training_dates[t]][step_ahead]

def y(s, t):
    return ground_truth[training_counties[s]][training_dates[t]]

w = np.full(K, 1 / K)
sigma = 100

for _ in range(50):
    w, sigma = EM_iteration(f, y, K, T, w, sigma)
print(w, sigma)

# # Store forecast, ground truth in a nicer format in `flu_fct_processed.csv`
#
# out = open("../task1_data/flu_fct_processed.csv", "w", newline='')
# forecast_writer = csv.writer(out)
# forecast_writer.writerow(["raw_data"] + list(dates[start_index+t] for t in range(T)))
# for k in range(K):
#     forecast_writer.writerow([methods[k]] + list(str(f(k, t)) for t in range(T)))
# forecast_writer.writerow(["gtruth"] + list(str(y(t)) for t in range(T)))
# out.close()
#
# # perform EM and store values to `ensemble_parameters.csv`
#
# iters = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# prev = 0
#
# out = open("../task1_data/ensemble_parameters.csv", "w", newline='')
# parameters_writer = csv.writer(out)
# parameters_writer.writerow(["EM_iters", "sigma"] + methods)
# for curr in iters:
#     for _ in range(curr - prev):
#         EM_iteration()
#     prev = curr
#     parameters_writer.writerow([str(curr), str(sigma)] + list(map(str, w)))
#     print(w)
#     print(sigma)
# out.close()
#
# # Store regression parameters
#
# out = open("../task1_data/regression_parameters.csv", "w", newline='')
# parameters_writer = csv.writer(out)
# parameters_writer.writerow(["raw_data"] + methods)
# parameters_writer.writerow(["a"] + list(str(regression_parameters[k][0]) for k in range(K)))
# parameters_writer.writerow(["b"] + list(str(regression_parameters[k][1]) for k in range(K)))
# out.close()
#
