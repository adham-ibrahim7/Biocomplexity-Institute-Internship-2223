import itertools
from collections import defaultdict

import numpy as np

from src.ensemble_forecast import EnsembleForecast


class Shapley:
    def __init__(self, all_data_df, county, training_methods, training_dates, horizon, step_ahead, print_status=False):
        self.all_data_df = all_data_df
        self.county = county
        self.training_methods = training_methods
        self.training_dates = training_dates
        self.horizon = horizon
        self.step_ahead = step_ahead
        self.print_status = print_status

        self.K = len(self.training_methods)
        self.T = len(self.training_dates)

        self.cache = {}

    def payoff(self, methods):
        if len(methods) == 0:
            return 0

        # print(methods)

        key = tuple(sorted(methods))
        if key in self.cache:
            # print("IN CACHE :)")
            return self.cache[key]
            pass

        forecast = EnsembleForecast(self.all_data_df, self.county, methods, self.training_dates, self.horizon, self.step_ahead)
        mean = forecast.get_mean(self.horizon)

        value = abs(mean - forecast.ground_truth[self.horizon])
        self.cache[key] = value

        return value

    def exact_shapley(self):
        shapley = defaultdict(float)

        permutations = itertools.permutations(self.training_methods.copy())

        count = 0

        for curr_permutation in permutations:
            count += 1
            # print("{}/{}".format(count, np.math.factorial(len(self.training_methods))))

            values = []
            for k in range(0, self.K+1):
                values.append(self.payoff(curr_permutation[:k]))

            for k in range(0, self.K):
                difference = values[k+1] - values[k]
                shapley[curr_permutation[k]] += difference

        for method in shapley:
            shapley[method] /= count

        return list(shapley[method] for method in self.training_methods)

    def approximate_shapley(self, num_permutations=1000):
        shapley = defaultdict(float)

        curr_permutation = self.training_methods.copy()

        count = 0

        for _ in range(num_permutations):
            count += 1
            # print("{}/{}".format(count, num_permutations))

            curr_permutation = np.random.permutation(curr_permutation)

            for k in range(0, self.K):
                difference = self.payoff(curr_permutation[:k+1]) - self.payoff(curr_permutation[:k])
                shapley[curr_permutation[k]] += difference / num_permutations

        return list(shapley[method] for method in self.training_methods)