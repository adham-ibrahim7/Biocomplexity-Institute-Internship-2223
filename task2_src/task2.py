import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.integrate import quad as integral
import timeit

# TODO: Create proper docstrings for each method

class EnsembleForecast:
    def __init__(self, all_data_df, training_counties, training_methods, training_dates, step_ahead):
        self.all_data_df = all_data_df
        self.training_counties = training_counties
        self.training_methods = training_methods
        self.training_dates = training_dates
        self.step_ahead = step_ahead

        self.forecasts, self.ground_truth = self.get_data()
        self.regression_parameters = self.get_regression_parameters()

        self.S = len(self.training_counties)
        self.K = len(self.training_methods)
        self.T = len(self.training_dates)

        self.weights, self.uncalibrated_sigma = self.EM(init_sigma=100, iters=10)

        self.fct_date = self.training_dates[-1]

        print("uncalibrated sigma", self.uncalibrated_sigma)
        self.plot_pdf(training_counties[0], self.fct_date, self.uncalibrated_sigma, np.arange(10000, 16000))

        self.calibrated_sigma = self.get_calibrated_sigma(self.fct_date, 0, 400, 5)

        print("calibrated sigma", self.calibrated_sigma)
        self.plot_pdf(training_counties[0], self.fct_date, self.calibrated_sigma, np.arange(12400, 13000))

    def get_data(self):
        print("BEGIN INPUTTING RAW DATA.")

        forecasts = {}
        ground_truth = {}

        for county in self.training_counties:
            county_df = self.all_data_df.query("cnty == '{}'".format(county))
            forecasts[county] = {}
            ground_truth[county] = {}
            for method in self.training_methods:
                county_method_df = county_df.query("method == '{}'".format(method))
                forecasts[county][method] = {}
                for date in self.training_dates:
                    target_df = county_method_df.query(
                        "horizon == '{}' and step_ahead == '{}'".format(date, self.step_ahead))
                    if not target_df.empty:
                        fct_mean = target_df.at[target_df.index[0], 'fct_mean']
                        forecasts[county][method][date] = fct_mean

                    # TODO: Read ground truth only once outside this class.
                    if date not in ground_truth[county]:
                        temp = county_method_df.query("fct_date == @date")
                        if temp.empty:
                            continue
                        ground_truth[county][date] = temp.at[temp.index[0], 'true']

        print("DONE INPUTTING RAW DATA.")

        return forecasts, ground_truth

    def get_regression_parameters(self):
        regression_parameters = {}

        for method in self.training_methods:
            X = []
            Y = []
            for county in self.training_counties:
                # print(method.ljust(20), end=' ')
                # for day in training_dates:
                #     print('T' if day in valid_dates[method] else ' ', end='')
                # print()

                X += list(map(lambda x: self.forecasts[county][method][x], self.training_dates))
                Y += list(map(lambda x: self.ground_truth[county][x], self.training_dates))

            slope, intercept, _, _, _ = stats.linregress(X, Y)
            regression_parameters[method] = (intercept, slope)
            # print(method, regression_parameters[method])

        return regression_parameters

    def EM(self, init_sigma=100, iters=10):
        def f(s, k, t):
            return self.forecasts[self.training_counties[s]][self.training_methods[k]][self.training_dates[t]]

        def y(s, t):
            return self.ground_truth[self.training_counties[s]][self.training_dates[t]]

        init_w = np.full(self.K, 1 / self.K)

        w = init_w
        sigma = init_sigma

        print("BEGINNING EM ALGORITHM.")

        for _ in range(iters):
            q = np.zeros((self.S, self.K, self.T))
            for s in range(self.S):
                for k in range(self.K):
                    a, b = self.regression_parameters[self.training_methods[k]]
                    for t in range(self.T):
                        q[s][k][t] = w[k] * stats.norm(a + b * f(s, k, t), sigma).pdf(y(s, t))
            z = np.zeros((self.S, self.K, self.T))
            for s in range(self.S):
                for k in range(self.K):
                    for t in range(self.T):
                        z[s][k][t] = q[s][k][t] / sum(q[s][j][t] for j in range(self.K))

            w = np.zeros(self.K)
            for k in range(self.K):
                w[k] = np.mean(z[:, k, :])

            variance = 0
            for s in range(self.S):
                for t in range(self.T):
                    for k in range(self.K):
                        variance += z[s][k][t] * np.square(y(s, t) - f(s, k, t))
            variance /= (self.S * self.T)

            sigma = np.sqrt(variance)

        print("EM ALGORITHM COMPLETE.")

        return w, sigma

    def sample(self, county, date, sigma):
        k = np.random.choice(list(range(self.K)), p=self.weights)
        a, b = self.regression_parameters[self.training_methods[k]]
        mean = a + b * self.forecasts[county][self.training_methods[k]][date]
        return np.random.normal(mean, sigma)

    def get_confidence_interval(self, county, date, sigma, CI_size, n_samples=1000):
        samples = np.empty(n_samples)
        for i in range(n_samples):
            samples[i] = self.sample(county, date, sigma)

        samples = sorted(samples)

        start_index = int(n_samples * (.5 * (1 - CI_size)))
        end_index = n_samples - start_index
        return samples[start_index], samples[end_index]

    def ensemble_cdf(self, county, date, sigma, X):
        cdf = 0
        for k in range(self.K):
            a, b = self.regression_parameters[self.training_methods[k]]
            cdf += self.weights[k] * stats.norm.cdf(X, a + b * self.forecasts[county][self.training_methods[k]][date], sigma)
        return cdf

    # TODO: This is very slow, even with a ternary search. Look into other methods.
    def CRPS(self, date, sigma):
        cdf = lambda county: lambda t: self.ensemble_cdf(county, date, sigma, t)

        score = 0
        for county in self.training_counties:
            def f_1(t):
                y = cdf(county)(t)
                return y * y

            def f_2(t):
                y = cdf(county)(t)
                return (1 - y) * (1 - y)

            # x_axis = np.arange(10000, 16000, 20)
            # plt.plot(x_axis, cdf(county)(x_axis))
            # plt.show()

            true_value = self.ground_truth[county][date]
            score += integral(f_1, -np.inf, true_value)[0] + integral(f_2, true_value, np.inf)[0]

        return score

    def get_calibrated_sigma(self, date, lo, hi, tolerance):
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
            if self.CRPS(date, mid_left) < self.CRPS(date, mid_right):
                hi = mid_right
            else:
                lo = mid_left

        print("CALIBRATION COMPLETE, ITERS=", count)

        return lo

    def filter_valid_dates(self, county, method, dates):
        return np.array(list(filter(lambda t: t in self.forecasts[county][method], dates)))

    def plot_forecasts(self, plt_counties, plt_methods, plt_dates):
        for county in plt_counties:
            plt.clf()
            plt.plot(plt_dates, list(map(lambda x: self.ground_truth[county][x], plt_dates)), color='black')
            for method in plt_methods:
                valid_dates = self.filter_valid_dates(county, method, plt_dates)
                plt.plot(valid_dates, list(map(lambda x: self.forecasts[county][method][x], valid_dates)))
            # plt.xticks(list(all_dates[i] for i in range(0, len(all_dates), 10)), rotation=30)
            plt.legend(labels=['gtruth'] + plt_methods)
            plt.title(county + " forecasts")
            plt.gcf().subplots_adjust(bottom=0.2)
            plt.show()

    def plot_forecasts_error(self, plt_counties, plt_methods, plt_dates):
        for county in plt_counties:
            plt.clf()
            for method in plt_methods:
                valid_dates = self.filter_valid_dates(county, method, plt_dates)
                plt.plot(valid_dates,
                         list(map(lambda x: self.forecasts[county][method][x] / self.ground_truth[county][x], valid_dates)))
            # plt.xticks(list(all_dates[i] for i in range(0, len(all_dates), 10)), rotation=30)
            plt.legend(labels=plt_methods)
            plt.title(county + " error")
            plt.gcf().subplots_adjust(bottom=0.2)
            plt.show()

    def ensemble_pdf(self, county, date, curr_sigma, x_axis):
        pdf = 0
        for k in range(K):
            a, b = self.regression_parameters[training_methods[k]]
            adjusted_fct_mean = a + b * self.forecasts[county][training_methods[k]][date]
            pdf += self.weights[k] * stats.norm.pdf(x_axis, adjusted_fct_mean, curr_sigma)
        return pdf

    def plot_pdf(self, county, date, sigma, x_axis):
        plt.plot(x_axis, self.ensemble_pdf(county, date, sigma, x_axis))

        for u in [.5, .95]:
            l, r = self.get_confidence_interval(county, date, sigma, u)
            plt.axvline(x=l, color="black", linestyle="dashed")
            plt.axvline(x=r, color="black", linestyle="dashed")
        plt.axvline(x=self.ground_truth[county][date], color="black")

        plt.show()

################################################################################################
# MAIN CODE
################################################################################################

if __name__ == "__main__":
    in_filename = "../task2_data/top_10_pop_all_step_ahead.csv"

    all_data = pd.read_csv(in_filename, dtype={"cnty": "str"})

    all_counties = all_data.cnty.unique()
    all_methods = all_data.method.unique()
    all_dates = all_data.horizon.unique()

    for county in all_counties[:1]:
        training_counties = [county]
        print("PROCESSING", county)

        step_ahead = '1-step_ahead'
        training_methods = ['AR', 'ARIMA', 'AR_spatial', 'ENKF', 'PatchSim_adpt']

        S = len(training_counties)
        K = len(training_methods)
        T = 15
        training_dates = all_dates[1:T+1]

        forecast = EnsembleForecast(all_data, training_counties, training_methods, training_dates, step_ahead)