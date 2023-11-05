from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


@dataclass
class State:
    time: int
    variance: npt.NDArray[np.float64]
    correlation: npt.NDArray[np.float64]
    weight: npt.NDArray[np.float64]


@dataclass
class States:
    time: int
    variances: npt.NDArray[np.float64]
    correlations: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
