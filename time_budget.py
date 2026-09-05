"""Cooperative elapsed-time budgets shared by all stages of a method."""

import math
import time


class BudgetExpired(Exception):
    """The method should stop at its current safe checkpoint."""


class TimeBudget:
    def __init__(self, time_limit=None):
        if time_limit is not None:
            time_limit = float(time_limit)
            if math.isnan(time_limit) or time_limit < 0:
                raise ValueError("time_limit must be nonnegative and not NaN")
            if math.isinf(time_limit):
                time_limit = None
        self.limit = time_limit
        self.start = time.perf_counter()

    def elapsed(self):
        return time.perf_counter() - self.start

    def remaining(self):
        if self.limit is None:
            return None
        return max(0.0, self.limit - self.elapsed())

    def expired(self):
        remaining = self.remaining()
        return remaining is not None and remaining <= 0.0

    def check(self):
        if self.expired():
            raise BudgetExpired

    def apply_to(self, model):
        """Check and refresh the solver allowance immediately before optimize()."""
        remaining = self.remaining()
        if remaining is not None:
            if remaining <= 0.0:
                raise BudgetExpired
            model.Params.TimeLimit = remaining
