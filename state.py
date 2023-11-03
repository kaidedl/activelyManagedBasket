from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


@dataclass
class State:
    time: int
    spot: npt.NDArray[np.float64]
    variance: npt.NDArray[np.float64]
    correlation: npt.NDArray[np.float64]
    weight: npt.NDArray[np.float64]


@dataclass
class States:
    time: int
    spots: npt.NDArray[np.float64]
    variances: npt.NDArray[np.float64]
    correlations: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
