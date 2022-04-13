from dataclasses import dataclass
import numpy as np


@dataclass
class Bounds:
    ub: np.ndarray
    lb: np.ndarray

    def radius(self):
        return np.max(self.ub - self.lb)

    def extend_tr(self, center, radius):
        self.extend(center + radius)
        self.extend(center - radius)
        return self

    def sample(self):
        if self.ub is None or self.lb is None:
            raise Exception()
        
        return self.lb + np.multiply(
            np.random.random(len(self.ub)), self.ub - self.lb
        )

    def extend(self, x):
        if self.ub is None:
            self.ub = np.copy(x)
        if self.lb is None:
            self.lb = np.copy(x)

        for i in range(len(x)):
            if x[i] > self.ub[i]:
                self.ub[i] = x[i]
            if x[i] < self.lb[i]:
                self.lb[i] = x[i]
        return self

    def buffer(self, amount):
        self.ub += amount
        self.lb -= amount
        return self

    def expand(self, factor=1.2):
        ub = np.copy(self.ub)
        lb = np.copy(self.lb)
        for i in range(len(self.ub)):
            expansion = (factor - 1.0) * (ub[i] - lb[i])
            ub[i] = ub[i] + expansion
            lb[i] = lb[i] - expansion
        return Bounds(lb=lb, ub=ub)
