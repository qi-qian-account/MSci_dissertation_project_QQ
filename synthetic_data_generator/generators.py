import numpy as np
from .helpers import (
    _make_supports,
    _as_array,
    _sample_noise,
    _shuffle_outputs,
)

def generate_additive_biclusters(
    n_clusters=3,
    total_rows=1000,
    total_cols=100,
    row_fraction=0.1,
    col_fraction=0.1,
    contrast_level=3.0,
    row_overlap_ratio=0.15,
    col_overlap_ratio=0.15,
    background_noise_std=1.0,
    row_effect_std=0.2,
    col_effect_std=0.2,
    within_block_noise_std=0.1,
    size_jitter=0.0,
    placement="diagonal",
    noise_distribution="gaussian",
    noise_df=5,
    signed_blocks=False,
    cell_signal_density=1.0,
    seed=42,
    dtype=np.float64,
):
    """
    Generate synthetic biclustering data using an additive Plaid-style model.

    Main controlled factors
    -----------------------
    n_clusters : number of biclusters.
    row_fraction : row support size as fraction of total rows.
    col_fraction : column support size as fraction of total columns.
    contrast_level : planted bicluster mean effect size.
    row_overlap_ratio : overlap between consecutive row supports.
    col_overlap_ratio : overlap between consecutive column supports.
    background_noise_std : ambient noise level.
    row_effect_std : standard deviation of row effects, relative to contrast_level.
    col_effect_std : standard deviation of column effects, relative to contrast_level.
    within_block_noise_std : cell-level noise inside biclusters, relative to contrast_level.
    cell_signal_density : fraction of cells inside each bicluster receiving signal.
                          Default 1.0 gives full rectangular biclusters.

    Returns
    -------
    result : dict
        Contains shuffled data, unshuffled data, ground truth indicators,
        signal matrix, noise matrix, permutations, and metadata.
    """
    rng = np.random.default_rng(seed)

    contrast = _as_array(
        contrast_level, n_clusters, dtype=float, name="contrast_level"
    )

    row_indicators, row_supports = _make_supports(
        total_size=total_rows,
        n_clusters=n_clusters,
        size_fraction=row_fraction,
        overlap_ratio=row_overlap_ratio,
        rng=rng,
        placement=placement,
        size_jitter=size_jitter,
    )

    col_indicators, col_supports = _make_supports(
        total_size=total_cols,
        n_clusters=n_clusters,
        size_fraction=col_fraction,
        overlap_ratio=col_overlap_ratio,
        rng=rng,
        placement=placement,
        size_jitter=size_jitter,
    )

    noise = _sample_noise(
        rng,
        shape=(total_rows, total_cols),
        noise_std=background_noise_std,
        noise_distribution=noise_distribution,
        df=noise_df,
    ).astype(dtype)

    signal = np.zeros((total_rows, total_cols), dtype=dtype)

    cluster_params = []

    for k in range(n_clusters):
        rows = row_supports[k]
        cols = col_supports[k]

        block_sign = rng.choice([-1.0, 1.0]) if signed_blocks else 1.0
        mu_k = block_sign * contrast[k]

        alpha = rng.normal(
            0.0,
            row_effect_std * abs(contrast[k]),
            size=len(rows),
        )

        beta = rng.normal(
            0.0,
            col_effect_std * abs(contrast[k]),
            size=len(cols),
        )

        block_noise = rng.normal(
            0.0,
            within_block_noise_std * abs(contrast[k]),
            size=(len(rows), len(cols)),
        )

        block = mu_k + alpha[:, None] + beta[None, :] + block_noise

        if not (0 < cell_signal_density <= 1):
            raise ValueError("cell_signal_density must be in (0, 1].")

        if cell_signal_density < 1.0:
            cell_mask = rng.random(size=block.shape) < cell_signal_density
            block = block * cell_mask

        signal[np.ix_(rows, cols)] += block

        cluster_params.append(
            {
                "cluster": k,
                "n_rows": len(rows),
                "n_cols": len(cols),
                "mu": mu_k,
                "contrast_level": contrast[k],
            }
        )

    data = signal + noise

    (
        shuffled_data,
        shuffled_row_indicators,
        shuffled_col_indicators,
        row_perm,
        col_perm,
    ) = _shuffle_outputs(data, row_indicators, col_indicators, rng)

    return {
        "shuffled_data": shuffled_data.astype(dtype),
        "data": data.astype(dtype),
        "signal_matrix": signal.astype(dtype),
        "noise_matrix": noise.astype(dtype),
        "true_row_indicators": row_indicators,
        "true_col_indicators": col_indicators,
        "shuffled_row_indicators": shuffled_row_indicators,
        "shuffled_col_indicators": shuffled_col_indicators,
        "row_perm": row_perm,
        "col_perm": col_perm,
        "row_supports": row_supports,
        "col_supports": col_supports,
        "cluster_params": cluster_params,
        "metadata": {
            "model": "additive",
            "n_clusters": n_clusters,
            "total_rows": total_rows,
            "total_cols": total_cols,
            "row_fraction": row_fraction,
            "col_fraction": col_fraction,
            "contrast_level": contrast_level,
            "row_overlap_ratio": row_overlap_ratio,
            "col_overlap_ratio": col_overlap_ratio,
            "background_noise_std": background_noise_std,
            "row_effect_std": row_effect_std,
            "col_effect_std": col_effect_std,
            "within_block_noise_std": within_block_noise_std,
            "size_jitter": size_jitter,
            "placement": placement,
            "noise_distribution": noise_distribution,
            "signed_blocks": signed_blocks,
            "cell_signal_density": cell_signal_density,
            "seed": seed,
        },
    }

def generate_multiplicative_biclusters(
    n_clusters=3,
    total_rows=1000,
    total_cols=100,
    row_fraction=0.1,
    col_fraction=0.1,
    contrast_level=3.0,
    row_overlap_ratio=0.15,
    col_overlap_ratio=0.15,
    background_noise_std=1.0,
    latent_variability=0.15,
    factor_background_std=0.0,
    size_jitter=0.0,
    placement="diagonal",
    noise_distribution="gaussian",
    noise_df=5,
    signed_rows=True,
    signed_factors=False,
    seed=42,
    dtype=np.float64,
):
    """
    Generate synthetic biclustering data using a multiplicative latent factor model.

    Model
    -----
    X = Lambda @ Z + E

    Main controlled factors
    -----------------------
    n_clusters : number of biclusters.
    row_fraction : row support size as fraction of total rows.
    col_fraction : column support size as fraction of total columns.
    contrast_level : expected planted product magnitude inside biclusters.
    row_overlap_ratio : overlap between consecutive row supports.
    col_overlap_ratio : overlap between consecutive column supports.
    background_noise_std : ambient noise level.
    latent_variability : variability of active Lambda and Z entries.
    factor_background_std : nonzero value gives weak global latent background.
                            Use 0.0 for clean sparse FABIA-style factors.

    Returns
    -------
    result : dict
        Contains shuffled data, unshuffled data, true factors, ground truth
        indicators, signal matrix, noise matrix, permutations, and metadata.
    """
    rng = np.random.default_rng(seed)

    contrast = _as_array(
        contrast_level, n_clusters, dtype=float, name="contrast_level"
    )

    if np.any(contrast <= 0):
        raise ValueError("contrast_level must be positive for multiplicative data.")

    row_indicators, row_supports = _make_supports(
        total_size=total_rows,
        n_clusters=n_clusters,
        size_fraction=row_fraction,
        overlap_ratio=row_overlap_ratio,
        rng=rng,
        placement=placement,
        size_jitter=size_jitter,
    )

    col_indicators, col_supports = _make_supports(
        total_size=total_cols,
        n_clusters=n_clusters,
        size_fraction=col_fraction,
        overlap_ratio=col_overlap_ratio,
        rng=rng,
        placement=placement,
        size_jitter=size_jitter,
    )

    Lambda = rng.normal(
        0.0,
        factor_background_std,
        size=(total_rows, n_clusters),
    ).astype(dtype)

    Z = rng.normal(
        0.0,
        factor_background_std,
        size=(n_clusters, total_cols),
    ).astype(dtype)

    cluster_params = []

    for k in range(n_clusters):
        rows = row_supports[k]
        cols = col_supports[k]

        # Split the desired product magnitude evenly across Lambda and Z.
        latent_mean = np.sqrt(contrast[k])

        row_values = latent_mean + rng.normal(
            0.0,
            latent_variability * latent_mean,
            size=len(rows),
        )

        col_values = latent_mean + rng.normal(
            0.0,
            latent_variability * latent_mean,
            size=len(cols),
        )

        if signed_rows:
            row_values *= rng.choice([-1.0, 1.0], size=len(rows))

        if signed_factors:
            factor_sign = rng.choice([-1.0, 1.0])
            row_values *= factor_sign

        Lambda[rows, k] = row_values
        Z[k, cols] = col_values

        cluster_params.append(
            {
                "cluster": k,
                "n_rows": len(rows),
                "n_cols": len(cols),
                "contrast_level": contrast[k],
                "latent_mean": latent_mean,
            }
        )

    signal = Lambda @ Z

    noise = _sample_noise(
        rng,
        shape=(total_rows, total_cols),
        noise_std=background_noise_std,
        noise_distribution=noise_distribution,
        df=noise_df,
    ).astype(dtype)

    data = signal + noise

    (
        shuffled_data,
        shuffled_row_indicators,
        shuffled_col_indicators,
        row_perm,
        col_perm,
    ) = _shuffle_outputs(data, row_indicators, col_indicators, rng)

    return {
        "shuffled_data": shuffled_data.astype(dtype),
        "data": data.astype(dtype),
        "signal_matrix": signal.astype(dtype),
        "noise_matrix": noise.astype(dtype),
        "Lambda": Lambda.astype(dtype),
        "Z": Z.astype(dtype),
        "true_row_indicators": row_indicators,
        "true_col_indicators": col_indicators,
        "shuffled_row_indicators": shuffled_row_indicators,
        "shuffled_col_indicators": shuffled_col_indicators,
        "row_perm": row_perm,
        "col_perm": col_perm,
        "row_supports": row_supports,
        "col_supports": col_supports,
        "cluster_params": cluster_params,
        "metadata": {
            "model": "multiplicative",
            "n_clusters": n_clusters,
            "total_rows": total_rows,
            "total_cols": total_cols,
            "row_fraction": row_fraction,
            "col_fraction": col_fraction,
            "contrast_level": contrast_level,
            "row_overlap_ratio": row_overlap_ratio,
            "col_overlap_ratio": col_overlap_ratio,
            "background_noise_std": background_noise_std,
            "latent_variability": latent_variability,
            "factor_background_std": factor_background_std,
            "size_jitter": size_jitter,
            "placement": placement,
            "noise_distribution": noise_distribution,
            "signed_rows": signed_rows,
            "signed_factors": signed_factors,
            "seed": seed,
        },
    }

def generate_null_dataset(
    null_type,
    total_rows=1000,
    total_cols=100,
    background_noise_std=1.0,
    seed=0,
    rank=3,
    global_strength=1.5,
):
    """
    Generate null/control datasets with no planted local biclusters.

    null_type:
        'gaussian_noise'
        'student_t_noise'
        'global_low_rank'
    """
    rng = np.random.default_rng(seed)

    if null_type == "gaussian_noise":
        data = rng.normal(
            0.0,
            background_noise_std,
            size=(total_rows, total_cols),
        )

    elif null_type == "student_t_noise":
        df = 5
        raw = rng.standard_t(df=df, size=(total_rows, total_cols))
        data = raw * background_noise_std / np.sqrt(df / (df - 2))

    elif null_type == "global_low_rank":
        U = rng.normal(0.0, 1.0, size=(total_rows, rank))
        V = rng.normal(0.0, 1.0, size=(rank, total_cols))

        low_rank_signal = global_strength * (U @ V) / np.sqrt(rank)
        noise = rng.normal(
            0.0,
            background_noise_std,
            size=(total_rows, total_cols),
        )

        data = low_rank_signal + noise

    else:
        raise ValueError(
            "null_type must be 'gaussian_noise', 'student_t_noise', or 'global_low_rank'."
        )

    row_perm = rng.permutation(total_rows)
    col_perm = rng.permutation(total_cols)

    shuffled_data = data[row_perm][:, col_perm]

    return {
        "shuffled_data": shuffled_data,
        "data": data,
        "row_perm": row_perm,
        "col_perm": col_perm,
    }
