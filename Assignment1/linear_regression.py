from dataclasses import dataclass, field
from typing import Any

from numpy.typing import NDArray


@dataclass(slots=True)
class LinearRegression:
    epochs: int = 10_000
    learning_rate: float = 4e-3
    weight: float = field(default=0.0, init=False)
    bias: float = field(default=0.0, init=False)

    @property
    def formula(self) -> str:
        return f"{self.weight:.2f}x{self.bias:+.2f}"

    def fit(self, X: NDArray[Any], y: NDArray[Any]) -> None:
        for _ in range(self.epochs):
            y_pred = self.predict(X)

            dw, db = self.compute_gradients(X, y, y_pred)
            self.update_parameters(dw, db)

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        return self.weight * X + self.bias

    def errors(self, X: NDArray[Any], y: NDArray[Any]) -> NDArray[Any]:
        return self.predict(X) - y

    def update_parameters(self, dw: float, db: float) -> None:
        self.weight -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    @staticmethod
    def compute_gradients(
        X: NDArray[Any], y: NDArray[Any], y_pred: NDArray[Any]
    ) -> tuple[float, float]:
        dl = y_pred - y
        dw = (dl * X).mean()
        db = dl.mean()
        return dw, db
