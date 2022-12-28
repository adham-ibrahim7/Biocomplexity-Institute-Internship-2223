import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.integrate import quad as integral
import timeit

# TODO: Create proper docstrings for each method
def get_data(all_data_df, counties, dates, methods, step_ahead):
    print("BEGIN INPUTTING RAW DATA.")

    forecasts: dict[str, dict[str, dict[str, float]]] = {}
    ground_truth: dict[str, dict[str, float]] = {}

    for county in counties:
        county_df = all_data_df.query("cnty == @county")
        forecasts[county] = {}
        ground_truth[county] = {}
        for method in methods:
            county_method_df = county_df.query("method == @method")
            forecasts[county][method] = {}
            for date in dates:
                county_method_date_df = county_method_df.query("horizon == @date")

                if not county_method_date_df.empty:
                    for i in range(len(county_method_date_df)):
                        curr_step_ahead = county_method_date_df.at[county_method_date_df.index[i], 'step_ahead']
                        fct_mean = county_method_date_df.at[county_method_date_df.index[i], 'fct_mean']
                        if curr_step_ahead == step_ahead:
                            forecasts[county][method][date] = fct_mean
                            break

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
def filter_valid_dates(forecasts, county, method, dates):
    return np.array(list(filter(lambda t: t in forecasts[county][method], dates)))

def plot(forecasts, ground_truth, plt_counties, plt_methods, plt_dates):
    for county in plt_counties:
        plt.clf()
        plt.plot(plt_dates, list(map(lambda x: ground_truth[county][x], plt_dates)), color='black')
        for method in plt_methods:
            valid_dates = filter_valid_dates(forecasts, county, method, plt_dates)
            plt.plot(valid_dates, list(map(lambda x: forecasts[county][method][x], valid_dates)))
        # plt.xticks(list(all_dates[i] for i in range(0, len(all_dates), 10)), rotation=30)
        plt.legend(labels=['gtruth'] + plt_methods)
        plt.title(county + " forecasts")
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.show()

def plot_error(forecasts, ground_truth, plt_counties, plt_methods, plt_dates):
    for county in plt_counties:
        plt.clf()
        for method in plt_methods:
            valid_dates = filter_valid_dates(forecasts, county, method, plt_dates)
            plt.plot(valid_dates, list(map(lambda x: forecasts[county][method][x] / ground_truth[county][x], valid_dates)))
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

            X += list(map(lambda x: forecasts[county][method][x], training_dates))
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

def EM(forecasts, ground_truth, training_counties, training_methods, training_dates, init_sigma=100, iters=10):
    def f(s, k, t):
        return forecasts[training_counties[s]][training_methods[k]][training_dates[t]]

    def y(s, t):
        return ground_truth[training_counties[s]][training_dates[t]]

    S = len(training_counties)
    K = len(training_methods)
    T = len(training_dates)

    init_w = np.full(K, 1 / K)

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

#TODO add all parameters to this
def ensemble_pdf(county, date, curr_sigma, X):
    pdf = 0
    for k in range(K):
        a, b = regression_parameters[training_methods[k]]
        adjusted_fct_mean = a + b * forecasts[county][training_methods[k]][date]
        pdf += w[k] * stats.norm.pdf(X, adjusted_fct_mean, curr_sigma)
    return pdf

def plot_pdf(county, date, sigma, x_axis):
    plt.plot(x_axis, ensemble_pdf(county, date, sigma, x_axis))

    # for u in [.5, .95]:
    #     l, r = get_CI(u, sigma)
    #     plt.axvline(x=l, color="black", linestyle="dashed")
    #     plt.axvline(x=r, color="black", linestyle="dashed")
    plt.axvline(x=ground_truth[county][date], color="black")

    plt.show()

################################################################################################
# GENERATE SAMPLES AND COMPUTE 50%, 95% CI
################################################################################################

def sample(county, date, sigma):
    k = np.random.choice(list(range(K)), p=w)
    a, b = regression_parameters[training_methods[k]]
    mean = a + b * forecasts[county][training_methods[k]][date]
    return np.random.normal(mean, sigma)

def get_confidence_interval(county, date, sigma, CI_size):
    n_samples = 1000
    samples = np.empty(n_samples)
    for i in range(n_samples):
        samples[i] = sample(county, date, sigma)

    samples = sorted(samples)

    start_index = int(n_samples * (.5 * (1 - CI_size)))
    end_index = n_samples - start_index
    return samples[start_index], samples[end_index]

################################################################################################
# CALIBRATION
################################################################################################

def ensemble_cdf(county, training_methods, date, regression_parameters, curr_sigma, X):
    cdf = 0
    for k in range(K):
        a, b = regression_parameters[training_methods[k]]
        cdf += w[k] * stats.norm.cdf(X, a + b * forecasts[county][training_methods[k]][date], curr_sigma)
    return cdf

# TODO: This is very slow, even with a ternary search. Look into other methods.
def CRPS(ground_truth, training_counties, training_methods, date, regression_parameters, sigma):
    cdf = lambda county: lambda t: ensemble_cdf(county, training_methods, date, regression_parameters, sigma, t)

    score = 0
    for county in training_counties:
        def f_1(t):
            y = cdf(county)(t)
            return y * y

        def f_2(t):
            y = cdf(county)(t)
            return (1 - y) * (1 - y)

        # x_axis = np.arange(10000, 16000, 20)
        # plt.plot(x_axis, cdf(county)(x_axis))
        # plt.show()

        true_value = ground_truth[county][date]
        score += integral(f_1, 0, true_value)[0] + integral(f_2, true_value, 30000)[0]

    return score


def get_calibrated_sigma(ground_truth, training_counties, training_methods, date, regression_parameters, lo, hi, tolerance):
    print("BEGIN CALIBRATION.")

    # best_sigma = 0
    # best_CRPS = 1000000
    #
    # for sigma in range:
    #     curr_CRPS = CRPS(ground_truth, training_counties, training_methods, date, regression_parameters, sigma)
    #     if curr_CRPS < best_CRPS:
    #         best_CRPS = curr_CRPS
    #         best_sigma = sigma

    count = 0
    while hi - lo > tolerance:
        count += 1

        mid_left = lo + (hi - lo) / 3
        mid_right = lo + (hi - lo) * 2 / 3
        if CRPS(ground_truth, training_counties, training_methods, date, regression_parameters, mid_left) < \
                CRPS(ground_truth, training_counties, training_methods, date, regression_parameters, mid_right):
            hi = mid_right
        else:
            lo = mid_left
    print("CALIBRATION COMPLETE, ITERS=", count)

    return lo

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

    forecasts, ground_truth = get_data(all_data, training_counties, training_dates, training_methods, step_ahead)

    regression_parameters = get_regression_parameters(forecasts, ground_truth, training_counties, training_methods, training_dates)

    # plt_counties = training_counties[:3]
    # plt_methods = ['AR_spatial']
    # plt_dates = all_dates[1:]

    # plot(plt_counties, plt_methods, plt_dates)
    # plot_error(plt_counties, plt_methods, plt_dates)

    w, uncalibrated_sigma = EM(forecasts, ground_truth, training_counties, training_methods, training_dates)

    fct_date = training_dates[-1]

    calibrated_sigma = get_calibrated_sigma(ground_truth, training_counties, training_methods, fct_date, regression_parameters, 0, 400, 5)

    for fct_cnty in training_counties:
        print(fct_cnty, "---------------------------------------------")
        true_value = ground_truth[fct_cnty][fct_date]

        def captured(fct_cnty, fct_date, sigma, u):
            lo, hi = get_confidence_interval(fct_cnty, fct_date, sigma, u)
            if lo < true_value < hi:
                start = "CAPTURED, MARGIN=" + str(min(true_value - lo, hi - true_value))
            else:
                start = "NOT CAPTURED"
            return start + "\n\t\t" + "({}, {})".format(lo, hi)

        print("BEFORE CALIBRATION: sigma=", uncalibrated_sigma)
        print("MIDDLE 50%:", captured(fct_cnty, fct_date, uncalibrated_sigma, .5))
        print("MIDDLE 95%:", captured(fct_cnty, fct_date, uncalibrated_sigma, .95))
        print("TRUE:", true_value)
        # plot_pdf(fct_cnty, fct_date, sigma, np.arange(10000, 16000, 20))

        print("AFTER CALIBRATION: sigma=", calibrated_sigma)
        print("MIDDLE 50%:", captured(fct_cnty, fct_date, calibrated_sigma, .5))
        print("MIDDLE 95%:", captured(fct_cnty, fct_date, calibrated_sigma, .95))
        print("TRUE:", true_value)
        # plot_pdf(fct_cnty, fct_date, calibrated_sigma, np.arange(10000, 16000, 20))