import numpy as np
from typing import Union

from biclustlib.algorithms import Plaid
from sklearn.metrics import r2_score
from pyspoc import ReducedStatistic


def fit_plaid_model(data: np.ndarray, num_biclusters: int):
    pld_model = Plaid(num_biclusters)
    pld_output = pld_model.run(data)
    return pld_output


def build_plaid_prediction_and_residual(data: np.ndarray, pld_output):
    mu_0 = np.mean(data)
    alpha_0 = np.mean(data, axis=1) - mu_0
    beta_0 = np.mean(data, axis=0) - mu_0

    predicted_matrix = mu_0 + alpha_0[:, np.newaxis] + beta_0[np.newaxis, :]
    residual_matrix = data - predicted_matrix

    for bc in pld_output.biclusters:
        rows = bc.rows
        cols = bc.cols

        residual_block = residual_matrix[np.ix_(rows, cols)]

        if residual_block.size > 0:
            mu_k = np.mean(residual_block)
            alpha_k = np.mean(residual_block, axis=1) - mu_k
            beta_k = np.mean(residual_block, axis=0) - mu_k

            block_effect = mu_k + alpha_k[:, np.newaxis] + beta_k[np.newaxis, :]

            predicted_matrix[np.ix_(rows, cols)] += block_effect
            residual_matrix[np.ix_(rows, cols)] -= block_effect

    return predicted_matrix, residual_matrix



def calculate_plaid_r2(data: np.ndarray, predicted_matrix: np.ndarray) -> float:
    y_true = data.flatten()
    y_pred = predicted_matrix.flatten()

    r2 = r2_score(y_true, y_pred)
    return float(max(0.0, r2))



def calculate_plaid_bic(data: np.ndarray, residual_matrix: np.ndarray, pld_output) -> float:
    total_rows, total_cols = data.shape
    n_cells = total_rows * total_cols

    # background params num
    n_parameters = 1 + (total_rows - 1) + (total_cols - 1)

    # bicluster layer params num
    for bc in pld_output.biclusters:
        num_cluster_rows = len(bc.rows)
        num_cluster_cols = len(bc.cols)

        if num_cluster_rows > 0 and num_cluster_cols > 0:
            n_parameters += 1 + (num_cluster_rows - 1) + (num_cluster_cols - 1)

    rss = np.sum(residual_matrix ** 2)

    if rss <= 0:
        return float(-np.inf)

    bic = n_cells * np.log(rss / n_cells) + n_parameters * np.log(n_cells)
    return float(bic)



def calculate_plaid_pwvet(data: np.ndarray, pld_output) -> float:

    total_weighted_quality = 0.0
    total_area = 0

    for bc in pld_output.biclusters:
        rows = np.asarray(bc.rows)
        cols = np.asarray(bc.cols)

        area = len(rows) * len(cols)

        if area == 0:
            continue

        total_area += area
        
        if len(rows) < 2 or len(cols) < 2: # degenerate bicluster
            q_k = 0.0

        else:
            block = data[np.ix_(rows, cols)]

            mu_cols = np.mean(block, axis=0)
            sigma_cols = np.std(block, axis=0)
            sigma_cols[sigma_cols == 0] = 1.0

            standardised_block = (block - mu_cols) / sigma_cols

            virtual_condition = np.mean(standardised_block, axis=1)
            error_matrix = np.abs(
                standardised_block - virtual_condition[:, np.newaxis]
            )

            ve_t = np.mean(error_matrix)

            if not np.isfinite(ve_t):
                q_k = 0.0
            else:
                q_k = 1.0 / (1.0 + ve_t)

        total_weighted_quality += area * q_k
    
    if total_area == 0:  # if all biclusters are degenerate
        return 1.0

    Q = total_weighted_quality / total_area
    p_wvet = 1.0 - Q

    return float(p_wvet)

class PlaidAllMetrics(ReducedStatistic):
    """
    Output order:
        self_1 = Plaid R2
        self_2 = Plaid BIC
        self_3 = Plaid pWVET
    """

    def __init__(self, num_biclusters: int, **kwargs):
        self.num_biclusters = num_biclusters
        super().__init__()

    @property
    def name(self) -> str:
        return "Plaid All Metrics"

    @property
    def identifier(self) -> str:
        return "plaid-all-metrics"

    @property
    def labels(self) -> list[str]:
        return ["biclustering", "plaid", "vector"]

    def compute(self, data: np.ndarray) -> Union[np.ndarray, float]:
        pld_output = fit_plaid_model(data, self.num_biclusters)

        predicted_matrix, residual_matrix = build_plaid_prediction_and_residual(
            data=data,
            pld_output=pld_output,
        )

        r2 = calculate_plaid_r2(
            data=data,
            predicted_matrix=predicted_matrix,
        )

        bic = calculate_plaid_bic(
            data=data,
            residual_matrix=residual_matrix,
            pld_output=pld_output,
        )

        pwvet = calculate_plaid_pwvet(
            data=data,
            pld_output=pld_output,
        )

        return np.array([r2, bic, pwvet], dtype=float)
