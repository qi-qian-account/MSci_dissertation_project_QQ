import numpy as np
from typing import Union

from pyspoc import ReducedStatistic

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter


class FABIAAllMetrics(ReducedStatistic):
    """
    Output order:
        self_1 = FABIA RMSE
        self_2 = FABIA AHS
        self_3 = FABIA SNR
    """

    def __init__(self, num_biclusters: int, thresZ: float = 0.5, **kwargs):
        self.num_biclusters = int(num_biclusters)
        self.thresZ = float(thresZ)
        self._fabia_r = None
        super().__init__()

    @property
    def name(self) -> str:
        return "FABIA All Metrics"

    @property
    def identifier(self) -> str:
        return "fabia-all-metrics"

    @property
    def labels(self) -> list[str]:
        return ["biclustering", "fabia", "rmse", "ahs", "snr", "vector"]

    def get_fabia_package(self):
        if self._fabia_r is None:
            self._fabia_r = importr("fabia")
        return self._fabia_r

    def row_mean_center(self, data: np.ndarray) -> np.ndarray:
        return data - np.mean(data, axis=1, keepdims=True) # mean centre each row

    def average_hoyer_sparseness(self,L_matrix: np.ndarray) -> float:
        n_rows, n_factors = L_matrix.shape
        sqrt_n = np.sqrt(n_rows)
        scores = []

        for k in range(n_factors):
            vector = L_matrix[:, k]
            l1_norm = np.linalg.norm(vector, ord=1)
            l2_norm = np.linalg.norm(vector, ord=2)

            if l2_norm == 0:
                score = 1.0
            else:
                score = (sqrt_n - (l1_norm / l2_norm)) / (sqrt_n - 1)

            scores.append(score)

        return float(np.mean(scores))

    def normalise_factors(self, L_matrix: np.ndarray, Z_matrix: np.ndarray):
        L_hat = np.zeros_like(L_matrix, dtype=float)
        Z_hat = np.zeros_like(Z_matrix, dtype=float)

        n_factors = Z_matrix.shape[0]
        for k in range(n_factors):
            z_k = Z_matrix[k, :]
            norm_factor = np.sqrt(np.mean(z_k ** 2))

            if norm_factor > 0:
                Z_hat[k, :] = z_k / norm_factor
                L_hat[:, k] = L_matrix[:, k] * norm_factor
            else:
                Z_hat[k, :] = z_k
                L_hat[:, k] = L_matrix[:, k]

        return L_hat, Z_hat

    def snr(self, L_hat: np.ndarray, Psi_vector: np.ndarray) -> float:
        trace_LLT = np.sum(L_hat ** 2)
        trace_Psi = np.sum(Psi_vector)

        if trace_Psi == 0:
            return float(np.inf)

        return float(trace_LLT / trace_Psi)

    def compute(self, data: np.ndarray) -> Union[np.ndarray, float]:
        data = np.asarray(data, dtype=float)
        centered_data = self.row_mean_center(data)

        fabia_r = self.get_fabia_package()

        # Fit FABIA
        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            result = fabia_r.fabia(centered_data, p=self.num_biclusters)

            L_matrix = np.array(ro.r("function(x) x@L")(result), dtype=float)
            Z_matrix = np.array(ro.r("function(x) x@Z")(result), dtype=float)
            Psi_vector = np.array(ro.r("function(x) x@Psi")(result), dtype=float)

        # RMSE
        X_pred = np.dot(L_matrix, Z_matrix)
        rmse = float(np.sqrt(np.mean((centered_data - X_pred) ** 2)))

        # AHS
        avg_hoyer_sparseness = self.average_hoyer_sparseness(L_matrix)

        # SNR
        L_hat, Z_hat = self.normalise_factors(L_matrix, Z_matrix)
        snr_value = self.snr(L_hat, Psi_vector)

        return np.array([rmse, avg_hoyer_sparseness, snr_value], dtype=float)
