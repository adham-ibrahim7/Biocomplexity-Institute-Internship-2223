import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.integrate import quad as integral
from stopwatch import Stopwatch

# TODO: Create proper docstrings for each method
class EnsembleForecast:
    def __init__(self, all_data_df, county, training_methods, training_dates, step_ahead):
        self.all_data_df = all_data_df
        # self.training_counties = training_counties
        self.county = county
        self.training_methods = training_methods
        self.training_dates = training_dates
        self.step_ahead = step_ahead

        self.forecasts, self.ground_truth = self.get_data()
        self.regression_parameters = self.get_regression_parameters()

        # self.S = len(self.training_counties)
        self.K = len(self.training_methods)
        self.T = len(self.training_dates)

        self.weights, self.uncalibrated_sigma = self.EM(init_sigma=100, max_iters=300)

        self.fct_date = self.training_dates[-1]

        print("uncalibrated sigma", self.uncalibrated_sigma)
        # self.plot_pdf(self.fct_date, self.uncalibrated_sigma, np.arange(0, 30000, 20))

        def captured(fct_date, sigma, u):
            lo, hi = self.get_confidence_interval(fct_date, sigma, u)
            true_value = self.ground_truth[fct_date]
            margin = min(abs(true_value - lo), abs(hi - true_value))
            if lo < true_value < hi:
                start = "CAPTURED, MARGIN={}".format(margin)
            else:
                start = "NOT CAPTURED, MARGIN={}".format(margin)
            return "{} ".format(fct_date) + start # + "\n\t\tinterval: ({}, {}), true: {}".format(lo, hi, true_value)

        # for date in training_dates:
        #     print(captured(date, self.uncalibrated_sigma, 0.5))

        # TODO: Is there a better way to choose bounds?
        self.calibrated_sigma = self.get_calibrated_sigma(lo=0, hi=self.uncalibrated_sigma * 3, tolerance=10)

        print("calibrated sigma", self.calibrated_sigma)
        # self.plot_pdf(self.fct_date, self.calibrated_sigma, np.arange(0, 30000, 20))
        #
        # for date in training_dates:
        #     print(captured(date, self.calibrated_sigma, 0.5))

    def get_mean(self, date):
        mean = 0
        for k in range(self.K):
            # TODO make regression_parameters a list not dict
            a, b = self.regression_parameters[training_methods[k]]
            mean += self.weights[k] * (a + b * self.forecasts[self.training_methods[k]][date])
        return mean

    def get_data(self):
        print("BEGIN INPUTTING RAW DATA.")

        forecasts: dict[str, dict[str, float]] = {}
        ground_truth: dict[str, float] = {}

        county_df = self.all_data_df.query("cnty == '{}'".format(self.county))
        for method in self.training_methods:
            county_method_df = county_df.query("method == '{}'".format(method))
            forecasts[method] = {}
            for date in self.training_dates:
                target_df = county_method_df.query(
                    "horizon == '{}' and step_ahead == '{}'".format(date, self.step_ahead))
                if not target_df.empty:
                    fct_mean = target_df.at[target_df.index[0], 'fct_mean']
                    forecasts[method][date] = fct_mean

                if date not in ground_truth:
                    temp = county_method_df.query("fct_date == @date")
                    if temp.empty:
                        continue
                    ground_truth[date] = temp.at[temp.index[0], 'true']

        print("DONE INPUTTING RAW DATA.")

        return forecasts, ground_truth

    def get_regression_parameters(self):
        regression_parameters = {}

        for method in self.training_methods:
            # for county in self.training_counties:
                # print(method.ljust(20), end=' ')
                # for day in training_dates:
                #     print('T' if day in valid_dates[method] else ' ', end='')
                # print()

            x = list(self.forecasts[method].values())
            y = list(self.ground_truth.values())

            slope, intercept, _, _, _ = stats.linregress(x, y)
            regression_parameters[method] = (intercept, slope)
            # print(method, regression_parameters[method])

        return regression_parameters

    def EM(self, init_sigma=100, max_iters=100, sigma_tolerance=0.01):
        def f(k, t):
            return self.forecasts[self.training_methods[k]][self.training_dates[t]]

        def y(t):
            return self.ground_truth[self.training_dates[t]]

        init_w = np.full(self.K, 1 / self.K)

        w = init_w
        sigma = init_sigma

        print("BEGINNING EM ALGORITHM.")

        for iter in range(max_iters):
            q = np.zeros((self.K, self.T))
            for k in range(self.K):
                a, b = self.regression_parameters[self.training_methods[k]]
                for t in range(self.T):
                    q[k][t] = w[k] * stats.norm(a + b * f(k, t), sigma).pdf(y(t))
            z = np.zeros((self.K, self.T))
            for k in range(self.K):
                for t in range(self.T):
                    z[k][t] = q[k][t] / sum(q[j][t] for j in range(self.K))

            w = np.zeros(self.K)
            for k in range(self.K):
                w[k] = np.mean(z[k, :])

            variance = 0
            for t in range(self.T):
                for k in range(self.K):
                    variance += z[k][t] * np.square(y(t) - f(k, t))
            variance /= self.T

            new_sigma = np.sqrt(variance)

            if abs(sigma - new_sigma) < sigma_tolerance:
                break

            sigma = new_sigma

            # if iter % 10 == 0:
            #     print(iter, sigma, w)

        print("EM ALGORITHM COMPLETE. ITERS={}".format(iter+1))

        return w, sigma

    def sample(self, date, sigma):
        k = np.random.choice(list(range(self.K)), p=self.weights)
        a, b = self.regression_parameters[self.training_methods[k]]
        mean = a + b * self.forecasts[self.training_methods[k]][date]
        return np.random.normal(mean, sigma)

    def get_confidence_interval(self, date, sigma, interval_size, n_samples=2500):
        samples = np.empty(n_samples)
        for i in range(n_samples):
            samples[i] = self.sample(date, sigma)

        samples = sorted(samples)

        start_index = int(n_samples * (.5 * (1 - interval_size)))
        end_index = n_samples - start_index
        return samples[start_index], samples[end_index]

    def ensemble_cdf(self, date, sigma, x_axis):
        cdf = 0
        for k in range(self.K):
            a, b = self.regression_parameters[self.training_methods[k]]
            cdf += self.weights[k] * stats.norm.cdf(x_axis, a + b * self.forecasts[self.training_methods[k]][date], sigma)
        return cdf

    # TODO: This is very slow, even with a ternary search. Look into other methods.
    def compute_CRPS(self, sigma):
        total_score = 0


        for date in training_dates:
            x_axis, _ = self.get_x_axis(date, sigma)

            cdf = lambda t: self.ensemble_cdf(date, sigma, t)

            def f_1(t):
                y = cdf(t)
                return y * y

            def f_2(t):
                y = cdf(t)
                return (1 - y) * (1 - y)

            # x_axis = np.arange(10000, 16000, 20)
            # plt.plot(x_axis, cdf(county)(x_axis))
            # plt.show()

            true_value = self.ground_truth[date]
            total_score += integral(f_1, x_axis[0], true_value)[0] + integral(f_2, true_value, x_axis[-1])[0]

        # print(sigma, total_score)

        return total_score

    def get_calibrated_sigma(self, lo=0, hi=1000, tolerance=10):
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
            if self.compute_CRPS(mid_left) < self.compute_CRPS(mid_right):
                hi = mid_right
            else:
                lo = mid_left

        calibrated_sigma = 0.5 * (lo + hi)

        print("CALIBRATION COMPLETE, ITERS={}".format(count))

        return calibrated_sigma

    def filter_valid_dates(self, method, dates):
        return np.array(list(filter(lambda date: date in self.forecasts[method], dates)))

    def plot_forecasts(self, plt_counties, plt_methods, plt_dates):
        for county in plt_counties:
            plt.clf()
            plt.plot(plt_dates, list(map(lambda date: self.ground_truth[date], plt_dates)), color='black')
            for method in plt_methods:
                valid_dates = self.filter_valid_dates(method, plt_dates)
                plt.plot(valid_dates, list(map(lambda x: self.forecasts[method][x], valid_dates)))
            # plt.xticks(list(all_dates[i] for i in range(0, len(all_dates), 10)), rotation=30)
            plt.legend(labels=['gtruth'] + plt_methods)
            plt.title(county + " forecasts")
            plt.gcf().subplots_adjust(bottom=0.2)
            plt.show()

    def plot_forecasts_error(self, plt_counties, plt_methods, plt_dates):
        for county in plt_counties:
            plt.clf()
            for method in plt_methods:
                valid_dates = self.filter_valid_dates(method, plt_dates)
                plt.plot(valid_dates,
                         list(map(lambda date: self.forecasts[method][date] / self.ground_truth[date], valid_dates)))
            # plt.xticks(list(all_dates[i] for i in range(0, len(all_dates), 10)), rotation=30)
            plt.legend(labels=plt_methods)
            plt.title(county + " error")
            plt.gcf().subplots_adjust(bottom=0.2)
            plt.show()

    def ensemble_pdf(self, date, curr_sigma, x_axis):
        pdf = 0
        for k in range(self.K):
            a, b = self.regression_parameters[self.training_methods[k]]
            adjusted_fct_mean = a + b * self.forecasts[self.training_methods[k]][date]
            pdf += self.weights[k] * stats.norm.pdf(x_axis, adjusted_fct_mean, curr_sigma)
        return pdf

    def get_x_axis(self, date, sigma, x_axis=np.arange(0, 30000), c=0.01):
        max_y = self.ensemble_pdf(date, sigma, self.get_mean(date))

        y = self.ensemble_pdf(date, sigma, x_axis)

        for i in range(len(x_axis)):
            if y[i] > c * max_y:
                break
        for j in reversed(range(len(x_axis))):
            if y[j] > c * max_y:
                break

        return x_axis[i:j], y[i:j]

    def plot_pdf(self, date, sigma, x_axis):
        # Autoscale the x_axis
        x_axis, y = self.get_x_axis(date, sigma, x_axis)

        plt.plot(x_axis, y)

        for u in [.5, .95]:
            l, r = self.get_confidence_interval(date, sigma, u)
            plt.axvline(x=l, color="black", linestyle="dashed")
            plt.axvline(x=r, color="black", linestyle="dashed")

        plt.axvline(x=self.ground_truth[date], color="black")

        plt.show()

################################################################################################
# MAIN CODE
################################################################################################

if __name__ == "__main__":
    in_filename = "../task2_data/top_10_pop_all_step_ahead.csv"

    all_data = pd.read_csv(in_filename, dtype={"cnty": "str"})

    all_counties = all_data.cnty.unique()
    all_methods = all_data.method.unique()
    all_dates = all_data.horizon.unique()[1:]

    for county in all_counties[:1]:
        print("PROCESSING", county)

        step_ahead = '2-step_ahead'
        training_methods = ['AR', 'ARIMA', 'AR_spatial', 'ENKF', 'PatchSim_adpt']

        T = 15
        training_dates = all_dates[0:T]

        forecast = EnsembleForecast(all_data, county, training_methods, training_dates, step_ahead)
