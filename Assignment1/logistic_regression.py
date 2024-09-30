from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy import floating
from numpy.typing import NDArray


def sigmoid(z: NDArray[Any]) -> NDArray[Any]:
    return (1.0 + np.exp(-z)) ** (-1)


def cross_entropy_loss(y: NDArray[Any], y_pred: NDArray[Any]) -> floating:
    return -(y * np.log(y_pred) + (1.0 - y) * np.log(1.0 - y_pred)).mean()


@dataclass(slots=True)
class LogisticRegression:
    epochs: int = 1_000
    learning_rate: float = 1e-1
    weights: NDArray[Any] = field(
        default_factory=lambda: np.array([]), init=False
    )
    bias: float = field(default=0.0, init=False)
    losses: list[floating] = field(default_factory=list, init=False)
    accuracies: list[floating] = field(default_factory=list, init=False)

    @property
    def formula(self) -> str:
        return (
            f"[{', '.join(f'{weight:.2f}' for weight in self.weights)}]*x"
            f"{self.bias:+.2f}"
        )

    def linear_transform(self, X: NDArray[Any]) -> NDArray[Any]:
        return X @ self.weights + self.bias

    def fit(self, X: NDArray[Any], y: NDArray[Any]) -> None:
        n = X.shape[1]
        self.weights = np.zeros(n)
        self.losses.clear()
        self.accuracies.clear()

        for _ in range(self.epochs):
            z = self.linear_transform(X)
            y_pred = sigmoid(z)

            dw, db = self.compute_gradients(X, y, y_pred)
            self.update_parameters(dw, db)

            loss = cross_entropy_loss(y, y_pred)
            accuracy = self.score(X, y)
            self.losses.append(loss)
            self.accuracies.append(accuracy)

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        z = self.linear_transform(X)
        y_pred = sigmoid(z)
        return y_pred >= 0.5

    def score(self, X: NDArray[Any], y: NDArray[Any]) -> floating:
        y_pred = self.predict(X)
        return (y_pred == y).mean()

    def update_parameters(self, dw: NDArray[Any], db: float) -> None:
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    @staticmethod
    def compute_gradients(
        X: NDArray[Any], y: NDArray[Any], y_pred: NDArray[Any]
    ) -> tuple[NDArray[Any], float]:
        m = X.shape[0]
        dl = y_pred - y
        dw = dl @ X / m
        db = dl.mean()
        return dw, db
