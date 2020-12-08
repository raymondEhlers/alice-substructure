from functools import reduce

import attr
import numpy as np


@attr.s
class AsymmetricErrors:
    low: np.ndarray = attr.ib()
    high: np.ndarray = attr.ib()


@attr.s
class ErrorInput:
    value: np.ndarray = attr.ib()
    error: np.ndarray = attr.ib()


def relative_error(*inputs: ErrorInput) -> np.ndarray:
    if len(inputs) == 0:
        raise ValueError("Must pass at least one ErrorInput")
    if len(inputs) > 1:
        relative_error_squared = reduce(lambda x, y: ((x.error / x.value) ** 2) + ((y.error / y.value) ** 2), inputs)  # type: ignore
    else:
        relative_error_squared = (inputs[0].error / inputs[0].value) ** 2
    return np.sqrt(relative_error_squared)
