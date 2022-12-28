import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import timeit

# TODO: Create proper docstrings for each method
def get_data(all_data_df, counties, dates, methods):
    print("BEGIN INPUTTING RAW DATA.")

    forecasts: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    ground_truth: dict[str, dict[str, float]] = {}

    for county in counties:
        county_df = all_data_df.query("cnty == @county")
        forecasts[county] = {}
        ground_truth[county] = {}
        for method in methods:
            county_method_df = county_df.query("method == @method")
            forecasts[county][method] = {}
            for date in dates:
                forecasts[county][method][date] = {}
                county_method_date_df = county_method_df.query("horizon == @date")

                if not county_method_date_df.empty:
                    for i in range(len(county_method_date_df)):
                        step_ahead = county_method_date_df.at[county_method_date_df.index[i], 'step_ahead']
                        fct_mean = county_method_date_df.at[county_method_date_df.index[i], 'fct_mean']
                        forecasts[county][method][date][step_ahead] = fct_mean

                # TODO: Is there a better way to input the ground truth
                if date not in ground_truth[county]:
                    temp = county_method_df.query("fct_date == @date")
                    if temp.empty:
                        continue
                    ground_truth[county][date] = temp.at[temp.index[0], 'true']

    print("DONE INPUTTING RAW DATA.")

    return forecasts, ground_truth

################################################################################################
# PLOTTING FOR DEBUGGING
################################################################################################

# Returns days filtered from dates when the method had a forecast output
def filter_valid_dates(county, method, dates):
    return np.array(list(filter(lambda t: step_ahead in forecasts[county][method][t], dates)))

def plot(plt_counties, plt_methods, plt_dates):
    for county in plt_counties:
        plt.clf()
        plt.plot(plt_dates, list(map(lambda x: ground_truth[county][x], plt_dates)), color='black')
        for method in plt_methods:
            valid_dates = filter_valid_dates(county, method, plt_dates)
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
            valid_dates = filter_valid_dates(county, method, plt_dates)
            plt.plot(valid_dates, list(map(lambda x: forecasts[county][method][x][step_ahead] / ground_truth[county][x], valid_dates)))
        # plt.xticks(list(all_dates[i] for i in range(0, len(all_dates), 10)), rotation=30)
        plt.legend(labels=plt_methods)
        plt.title(county + " error")
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.show()

################################################################################################
# LINEAR REGRESSION TO GET PARAMETERS a[k], b[k] FOR EACH METHOD
################################################################################################

def get_regression_parameters(forecasts, ground_truth, training_counties, training_methods, training_dates):
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

        slope, intercept, _, _, _ = stats.linregress(X, Y)
        regression_parameters[method] = (intercept, slope)
        # print(method, regression_parameters[method])

    return regression_parameters

################################################################################################
# EXPECTATION-MAXIMIZATION ALGORITHM TO FIND w, sigma
################################################################################################

def EM_iteration(f, y, S, K, T, w, sigma):
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

def EM(forecasts, ground_truth, training_counties, training_methods, training_dates, init_w, init_sigma, iters):
    def f(s, k, t):
        return forecasts[training_counties[s]][training_methods[k]][training_dates[t]][step_ahead]

    def y(s, t):
        return ground_truth[training_counties[s]][training_dates[t]]

    w = init_w
    sigma = init_sigma

    print("BEGINNING EM ALGORITHM.")

    for _ in range(iters):
        w, sigma = EM_iteration(f, y, S, K, T, w, sigma)

    print("EM ALGORITHM COMPLETE.")

    return w, sigma

################################################################################################
# PLOT PDF BEFORE CALIBRATION
################################################################################################

def ensemble_pdf(county, curr_sigma, X):
    pdf = 0
    for k in range(K):
        a, b = regression_parameters[training_methods[k]]
        adjusted_fct_mean = a + b * forecasts[county][training_methods[k]][training_dates[-1]][step_ahead]
        pdf += w[k] * stats.norm.pdf(X, adjusted_fct_mean, curr_sigma)
    return pdf

def plot_pdf(county, sigma, x_axis):
    plt.plot(x_axis, ensemble_pdf(county, sigma, x_axis))

    # for u in [.5, .95]:
    #     l, r = get_CI(u, sigma)
    #     plt.axvline(x=l, color="black", linestyle="dashed")
    #     plt.axvline(x=r, color="black", linestyle="dashed")
    plt.axvline(x=ground_truth[county][training_dates[-1]], color="black")

    plt.show()

################################################################################################
# GENERATE SAMPLES AND COMPUTE 50%, 95% CI
################################################################################################

def sample(county, sigma):
    k = np.random.choice(list(range(K)), p=w)
    a, b = regression_parameters[training_methods[k]]
    mean = a + b * forecasts[county][training_methods[k]][training_dates[-1]][step_ahead]
    return np.random.normal(mean, sigma)

def get_CI(county, sigma, CI_size):
    n_samples = 1000
    samples = np.empty(n_samples)
    for i in range(n_samples):
        samples[i] = sample(county, sigma)

    samples = sorted(samples)

    start_index = int(n_samples * (.5 * (1 - CI_size)))
    end_index = n_samples - start_index
    return samples[start_index], samples[end_index]

################################################################################################
# MAIN CODE
################################################################################################

if __name__ == "__main__":
    in_filename = "../task2_data/top_10_pop_all_step_ahead.csv"

    all_data = pd.read_csv(in_filename, dtype={"cnty": "str"})

    all_counties = all_data.cnty.unique()
    all_methods = all_data.method.unique()
    all_dates = all_data.horizon.unique()

    training_counties = all_counties[:5]

    step_ahead = '1-step_ahead'
    training_methods = ['AR', 'ARIMA', 'AR_spatial', 'ENKF', 'PatchSim_adpt']

    S = len(training_counties)
    K = len(training_methods)
    T = 15
    training_dates = all_dates[1:T+1]

    # TODO: only read in values for desired step-ahead, remove one dimension from forecasts dict
    forecasts, ground_truth = get_data(all_data, training_counties, training_dates, training_methods)

    regression_parameters = get_regression_parameters(forecasts, ground_truth, training_counties, training_methods, training_dates)

    # plt_counties = training_counties[:3]
    # plt_methods = ['AR_spatial']
    # plt_dates = all_dates[1:]

    # plot(plt_counties, plt_methods, plt_dates)
    # plot_error(plt_counties, plt_methods, plt_dates)

    w, sigma = EM(forecasts, ground_truth, training_counties, training_methods, training_dates, np.full(K, 1 / K), 100, 10)

    print(w, sigma)

    print("BEFORE CALIBRATION: sigma=", sigma)
    print("MIDDLE 50%:", get_CI(training_counties[0], sigma, .5))
    print("MIDDLE 95%:", get_CI(training_counties[0], sigma, .95))
    print("TRUE:", ground_truth[training_counties[0]][training_dates[-1]])
    plot_pdf(training_counties[0], sigma, np.arange(10000, 16000, 20))