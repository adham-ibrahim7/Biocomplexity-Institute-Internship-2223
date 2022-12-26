import pandas as pd
import timeit
import numpy as np
from scipy import stats

in_filename = "../task2_data/top_10_pop_all_step_ahead.csv"
all_data = pd.read_csv(in_filename, dtype={"cnty": "str"})

print("Processing raw data.")

counties = all_data.cnty.unique()
all_dates = all_data.horizon.unique()
methods = all_data.method.unique()

f: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
y: dict[str, dict[str, float]] = {}
for county in counties[:1]:
    all_county = all_data.query("cnty == @county")
    f[county] = {}
    y[county] = {}
    for method in methods:
        county_method = all_county.query("method == @method")
        f[county][method] = {}
        for date in all_dates:
            f[county][method][date] = {}
            county_method_date = county_method.query("horizon == @date")

            if not county_method_date.empty:
                for i in range(len(county_method_date)):
                    step_ahead = county_method_date.at[county_method_date.index[i], 'step_ahead']
                    fct_mean = county_method_date.at[county_method_date.index[i], 'fct_mean']
                    f[county][method][date][step_ahead] = fct_mean

            if date not in y[county]:
                temp = county_method.query("fct_date == @date")
                if temp.empty:
                    continue
                y[county][date] = temp.at[temp.index[0], 'true']

print("Done processing raw data.")

##################################################################################################

horizon = '2022-10-02'
step_ahead = '1-step_ahead'
fct_date = '2022-10-09'
county = '04013'

dates = {}

K = len(methods)

# method to perform linear regression of Y on X
def regression(X, Y):
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)

    U = X-X_mean
    V = Y-Y_mean
    b = np.dot(U, V) / np.dot(U, U)
    a = Y_mean - b * X_mean
    return a, b

# Compute a_k, b_k for each method: the regression parameters of the forecast's history on the ground truth

regression_parameters = {}
for method in methods:
    dates[method] = np.array(list(filter(lambda t: step_ahead in f[county][method][t], f[county][method].keys())))[1:]

    X = list(map(lambda x: f[county][method][x][step_ahead], dates[method]))
    Y = list(map(lambda x: y[county][x], dates[method]))

    regression_parameters[method] = regression(X, Y)
    print(method, regression_parameters[method])

# import matplotlib.pyplot as plt
#
# for method in methods:
#     plt.plot(dates[method], list(map(lambda x: f[county][method][x][step_ahead], dates[method])))
# plt.plot(y[county].keys(), y[county].values())
# plt.xticks(list(all_dates[i] for i in range(0, len(all_dates), 10)), rotation=30)
# plt.gcf().subplots_adjust(bottom=0.2)
# plt.show()

#
# # Expectation-maximization algorithm to find w, sigma
#
# z = np.full((K, T), 1 / K)
# w = np.full(K, 1 / K)
# sigma = 100
#
# def EM_iteration():
#     global w, z, sigma
#
#     q = np.zeros((K, T))
#     for k in range(K):
#         a, b = regression_parameters[k]
#         for t in range(T):
#             q[k][t] = w[k] * stats.norm(a + b * f(k, t), sigma).pdf(y(t))
#     temp = np.zeros((K, T))
#     for k in range(K):
#         for t in range(T):
#             temp[k][t] = q[k][t] / sum(q[j][t] for j in range(K))
#     z = temp
#
#     # print(z)
#
#     w = np.zeros(K)
#     for k in range(K):
#         w[k] = np.mean(z[k])
#
#     sigma = np.sqrt(1/T * sum(sum(z[k][t] * np.square(f(k, t)-y(t)) for k in range(K)) for t in range(T)))

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
