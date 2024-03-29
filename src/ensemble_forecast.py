import itertools
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats, integrate
import properscoring as ps
from stopwatch import Stopwatch


# TODO: Create proper docstrings for each method
class EnsembleForecast:
    def __init__(self, all_data_df, county, training_methods, training_dates, horizon, step_ahead, calibrate=False):
        self.all_data_df = all_data_df
        self.county = county
        self.training_methods = training_methods
        self.training_dates = training_dates
        self.horizon = horizon
        self.step_ahead = step_ahead
        # self.print_status = print_status

        self.all_dates = np.append(self.training_dates, self.horizon)

        self.forecasts, self.ground_truth = self.get_data()
        # print(self.ground_truth)
        self.regression_parameters = self.get_regression_parameters()

        self.K = len(self.training_methods)
        self.T = len(self.training_dates)

        self.weights, self.uncalibrated_sigma = self.BMA_expectation_maximization(max_iters=50)

        # stopwatch = Stopwatch()
        #
        # # TODO: Is there a better way to choose bounds?
        if calibrate:
            self.calibrated_sigma = self.get_calibrated_sigma(lo=0, hi=self.uncalibrated_sigma * 3, tolerance=100)
        #
        # stopwatch.stop()
        # print("Calibation time: ", stopwatch)

    def get_data(self):
        # if self.print_status:
        #     print("BEGIN INPUTTING RAW DATA.")

        forecasts: dict[str, dict[str, float]] = {}
        ground_truth: dict[str, float] = {}

        county_df = self.all_data_df.query("cnty == '{}'".format(self.county))

        for method in self.training_methods:
            county_method_df = county_df.query("method == '{}'".format(method))
            forecasts[method] = {}
            for date in self.all_dates:
                target_df = county_method_df.query("horizon == '{}' and step_ahead == '{}'".format(date, self.step_ahead))
                if not target_df.empty:
                    fct_mean = target_df.at[target_df.index[0], 'fct_mean']
                    if date not in ground_truth:
                        ground_truth[date] = target_df.at[target_df.index[0], 'true']
                else:
                    fct_mean = -1
                forecasts[method][date] = fct_mean

        # if self.print_status:
        #     print("DONE INPUTTING RAW DATA.")

        return forecasts, ground_truth

    def get_regression_parameters(self):
        regression_parameters = {}

        for method in self.training_methods:
            x = list(self.forecasts[method][date] for date in self.training_dates)
            y = list(self.ground_truth[date] for date in self.training_dates)

            slope, intercept, _, _, _ = stats.linregress(x, y)
            regression_parameters[method] = (intercept, slope)
            # print(method, regression_parameters[method])

        return regression_parameters

    def BMA_expectation_maximization(self, init_sigma=5000, max_iters=100, sigma_tolerance=1):
        def f(k, t):
            return self.forecasts[self.training_methods[k]][self.training_dates[t]]

        def y(t):
            return self.ground_truth[self.training_dates[t]]

        init_w = np.full(self.K, 1 / self.K)

        weights = init_w
        sigma = init_sigma

        # if self.print_status:
        #     print("BEGINNING EM ALGORITHM.")

        for iter in range(max_iters):
            # print(weights, sigma)

            q = np.zeros((self.K, self.T))
            for k in range(self.K):
                a, b = self.regression_parameters[self.training_methods[k]]
                for t in range(self.T):
                    q[k][t] = weights[k] * stats.norm(a + b * f(k, t), sigma).pdf(y(t))
                    # if q[k][t] == 0:
                    #     print(a + b * f(k, t), sigma, y(t))
            z = np.zeros((self.K, self.T))
            for k in range(self.K):
                for t in range(self.T):
                    z[k][t] = q[k][t] / sum(q[j][t] for j in range(self.K))

            # print(q, z)

            new_weights = np.zeros(self.K)
            for k in range(self.K):
                new_weights[k] = np.mean(z[k, :])

            variance = 0
            for t in range(self.T):
                for k in range(self.K):
                    variance += z[k][t] * np.square(y(t) - f(k, t))
            variance /= self.T

            new_sigma = np.sqrt(variance)

            # if abs(sigma - new_sigma) < sigma_tolerance:
            #     break

            sigma = new_sigma
            weights = new_weights

            # if iter % 10 == 0:
            #     print(iter, sigma, w)

        # if self.print_status:
        #     print("EM ALGORITHM COMPLETE. ITERS={}".format(iter+1))

        return weights, sigma

    def sample(self, date, sigma):
        k = np.random.choice(list(range(self.K)), p=self.weights)
        a, b = self.regression_parameters[self.training_methods[k]]
        mean = a + b * self.forecasts[self.training_methods[k]][date]
        return np.random.normal(mean, sigma)

    def get_confidence_interval(self, date, sigma, interval_size=0.5, n_samples=10000):
        samples = np.empty(n_samples)
        for i in range(n_samples):
            samples[i] = self.sample(date, sigma)

        samples = sorted(samples)

        start_index = int(n_samples * (.5 * (1 - interval_size)))
        end_index = n_samples - start_index

        m = 25
        left = np.mean(samples[start_index-m: start_index+m])
        right = np.mean(samples[end_index-m: end_index+m])

        return left, right

    def ensemble_cdf(self, date, sigma, x_axis):
        cdf = 0
        for k in range(self.K):
            a, b = self.regression_parameters[self.training_methods[k]]
            cdf += self.weights[k] * stats.norm.cdf(x_axis, a + b * self.forecasts[self.training_methods[k]][date], sigma)
        return cdf

    # TODO: This is very slow, even with a ternary search. Look into other methods.
    def compute_CRPS(self, sigma):
        total_score = 0

        for date in self.training_dates:
            x_axis, _ = self.get_pdf_x_axis(date, sigma)

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
            total_score += integrate.quad(f_1, x_axis[0], true_value)[0] + integrate.quad(f_2, true_value, x_axis[-1])[0]

        # print(sigma, total_score)

        return total_score

    def compute_CRPS_2(self, sigma):
        total_score = 0

        for date in self.training_dates:
            cdf = lambda t: self.ensemble_cdf(date, sigma, t)
            x_axis, _ = self.get_pdf_x_axis(date, sigma)
            ps.crps_quadrature(self.ground_truth[date], cdf_or_dist=cdf, xmin=x_axis[0], xmax=x_axis[-1], tol=0.01)

        return total_score

    def get_calibrated_sigma(self, lo=0, hi=1000, tolerance=10):
        # if self.print_status:
        #     print("BEGIN CALIBRATION.")

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

        # if self.print_status:
        #     print("CALIBRATION COMPLETE, ITERS={}".format(count))

        return calibrated_sigma

    def ensemble_pdf(self, date, curr_sigma, x_axis):
        pdf = 0
        for k in range(self.K):
            a, b = self.regression_parameters[self.training_methods[k]]
            adjusted_fct_mean = a + b * self.forecasts[self.training_methods[k]][date]
            pdf += self.weights[k] * stats.norm.pdf(x_axis, adjusted_fct_mean, curr_sigma)
        return pdf

    def get_pdf_x_axis(self, date, sigma, x_axis=None, c=0.01):
        if x_axis is None:
            MAX = abs(self.get_mean(date) * 5)
            x_axis = np.arange(-MAX, MAX, 20)

        max_y = self.ensemble_pdf(date, sigma, self.get_mean(date))

        y = self.ensemble_pdf(date, sigma, x_axis)

        i = 0
        while y[i] < c * max_y:
                i += 1

        j = len(y)-1
        while y[j] < c * max_y:
                j -= 1

        return x_axis[i:j], y[i:j]

    def plot_pdf(self, date, sigma, x_axis=None, show=True, save_to=None):
        plt.clf()
        # Autoscale the x_axis
        x_axis, y = self.get_pdf_x_axis(date, sigma, x_axis)

        plt.plot(x_axis, y)

        for u in [.5, .95]:
            l, r = self.get_confidence_interval(date, sigma, u)
            plt.axvline(x=l, color="black", linestyle="dashed")
            plt.axvline(x=r, color="black", linestyle="dashed")

        plt.axvline(x=self.ground_truth[date], color="black")
        plt.title("cnty={}, horizon={},\n step_ahead={}, lead_time={}".format(self.county, self.horizon, self.step_ahead, self.T))

        if save_to is not None:
            plt.savefig(save_to)

        if show:
            plt.show()

    def filter_valid_dates(self, method, dates):
        return np.array(list(filter(lambda date: date in self.forecasts[method], dates)))

    def plot_forecasts(self, show=True, save_to=None):
        plt.clf()

        plt.plot(self.training_dates, list(self.ground_truth[date] for date in self.training_dates), color='black')
        for method in self.training_methods:
            # valid_dates = self.filter_valid_dates(method, plt_dates)
            plt.plot(self.training_dates, list(self.get_bias_corrected_forecast(method, date) for date in self.training_dates))

        plt.xticks(self.all_dates, rotation=70)
        plt.legend(labels=['gtruth'] + self.training_methods)
        plt.title("cnty={}, horizon={},\n step_ahead={}, lead_time={}".format(self.county, self.horizon, self.step_ahead, self.T))
        plt.gcf().subplots_adjust(bottom=0.25)

        if save_to is not None:
            plt.savefig(save_to)

        if show:
            plt.show()

    # def plot_forecasts_error(self, plt_counties, plt_methods, plt_dates):
    #     for county in plt_counties:
    #         plt.clf()
    #         for method in plt_methods:
    #             valid_dates = self.filter_valid_dates(method, plt_dates)
    #             plt.plot(valid_dates,
    #                      list(map(lambda date: self.forecasts[method][date] / self.ground_truth[date], valid_dates)))
    #         # plt.xticks(list(all_dates[i] for i in range(0, len(all_dates), 10)), rotation=30)
    #         plt.legend(labels=plt_methods)
    #         plt.title(county + " error")
    #         plt.gcf().subplots_adjust(bottom=0.2)
    #         plt.show()

    def get_bias_corrected_forecast(self, method, date):
        a, b = self.regression_parameters[method]
        return a + b * self.forecasts[method][date]

    def get_mean(self, date):
        mean = 0
        for k in range(self.K):
            mean += self.weights[k] * self.get_bias_corrected_forecast(self.training_methods[k], date)
        return mean

    def get_weights(self):
        weights_dict = {}
        for k in range(self.K):
            weights_dict[self.training_methods[k]] = self.weights[k]
        return weights_dict

    def get_mae(self):
        mae = {}
        for method in self.training_methods:
            mae[method] = abs(self.ground_truth[self.horizon] - self.forecasts[method][self.horizon])
            # mae[method] = abs(self.ground_truth[self.horizon] - self.get_bias_corrected_forecast(method, self.horizon))
        return mae

    # def get_shapley_weights(self):
    #     shapley_value = self.approximate_shapley(num_permutations=10000,
    #                                                 payoff=self.mae_payoff)
    #
    #     weights = np.zeros(self.K)
    #
    #     sum = np.sum(list(shapley_value[method] for method in shapley_value))
    #     for k in range(self.K):
    #         weights[k] = shapley_value[self.training_methods[k]] / sum
    #
    #     return weights

    def mae_payoff(self, methods):
        if len(methods) == 0:
            return 0

        total_error = 0

        for method in methods:
            error = self.get_bias_corrected_forecast(method, self.horizon) - self.ground_truth[self.horizon]
            total_error += abs(error)

        return total_error / len(methods)

    # def squared_error_payoff(self, methods):
    #     if len(methods) == 0:
    #         return 0
    #
    #     total_error = 0
    #
    #     for method in methods:
    #         for date in self.training_dates:
    #             error = self.get_bias_corrected_forecast(method, date) - self.ground_truth[date]
    #             total_error += error * error
    #
    #     return total_error / (self.K * self.T)

    def exact_shapley(self, payoff=None):
        if payoff is None:
            raise Exception("Must provide a payoff function to exact_shapley()")

        shapley = defaultdict(float)

        permutations = itertools.permutations(self.training_methods.copy())

        count = 0

        for curr_permutation in permutations:
            count += 1
            for k in range(0, self.K):
                difference = payoff(curr_permutation[:k+1]) - payoff(curr_permutation[:k])
                shapley[curr_permutation[k]] += difference

        for method in shapley:
            shapley[method] /= count

        return list(shapley[method] for method in self.training_methods)

    def approximate_shapley(self, num_permutations=1000, payoff=None):
        if payoff is None:
            raise Exception("Must provide a payoff function to approximate_shapley()")

        shapley = defaultdict(float)

        curr_permutation = self.training_methods.copy()

        for _ in range(num_permutations):
            curr_permutation = np.random.permutation(curr_permutation)

            for k in range(0, self.K):
                difference = payoff(curr_permutation[:k+1]) - payoff(curr_permutation[:k])
                shapley[curr_permutation[k]] += difference / num_permutations

        return list(shapley[method] for method in self.training_methods)

