import numpy as np
import pandas as pd

def _as_array(x, n, dtype=float, name="parameter"):
    """
    Convert scalar or sequence input to an array of length n.
    """
    if np.isscalar(x):
        return np.full(n, x, dtype=dtype)

    arr = np.asarray(x, dtype=dtype)
    if arr.shape[0] != n:
        raise ValueError(f"{name} must be either scalar or length {n}.")
    return arr

def _sample_noise(rng, shape, noise_std=1.0, noise_distribution="gaussian", df=5):
    """
    Generate background noise.
    """
    if noise_distribution == "gaussian":
        return rng.normal(0.0, noise_std, size=shape)

    if noise_distribution == "laplace":
        return rng.laplace(0.0, noise_std / np.sqrt(2), size=shape)

    if noise_distribution == "student_t":
        if df <= 2:
            raise ValueError("Student-t noise requires df > 2 for finite variance.")
        raw = rng.standard_t(df=df, size=shape)
        return raw * noise_std / np.sqrt(df / (df - 2))

    raise ValueError("noise_distribution must be 'gaussian', 'laplace', or 'student_t'.")

def _make_supports(
    total_size,
    n_clusters,
    size_fraction=0.1,
    overlap_ratio=0.0,
    rng=None,
    placement="diagonal",
    size_jitter=0.0,
):
    """
    Construct row or column supports for biclusters.

    Parameters
    ----------
    total_size : int
        Number of rows or columns.
    n_clusters : int
        Number of biclusters.
    size_fraction : float or sequence of floats
        Fraction of rows/columns assigned to each bicluster.
    overlap_ratio : float
        Desired overlap area ratio between consecutive biclusters.
    placement : {'diagonal', 'random'}
        'diagonal' gives contiguous supports before shuffling.
        'random' gives randomly placed supports.
    size_jitter : float
        Random multiplicative jitter in cluster sizes.
        Example: 0.2 means sizes are multiplied by Uniform(0.8, 1.2).

    Returns
    -------
    indicators : ndarray, shape (total_size, n_clusters)
        Binary membership matrix.
    supports : list of ndarray
        List of index arrays, one per bicluster.
    """
    if rng is None:
        rng = np.random.default_rng()

    if not (0 <= overlap_ratio < 1):
        raise ValueError("overlap_ratio must be in [0, 1).")

    fractions = _as_array(size_fraction, n_clusters, dtype=float, name="size_fraction")

    if np.any(fractions <= 0) or np.any(fractions > 1):
        raise ValueError("All size fractions must be in (0, 1].")

    sizes = np.maximum(1, np.round(fractions * total_size).astype(int))

    if size_jitter > 0:
        jitter = rng.uniform(1 - size_jitter, 1 + size_jitter, size=n_clusters)
        sizes = np.maximum(1, np.round(sizes * jitter).astype(int))

    if np.any(sizes > total_size):
        raise ValueError("A requested cluster size is larger than the dimension.")

    indicators = np.zeros((total_size, n_clusters), dtype=np.int8)
    supports = []

    if placement == "diagonal":
        start = 0

        for k in range(n_clusters):
            size = sizes[k]

            if k > 0:
                previous_size = sizes[k - 1]
                overlap = int(round(overlap_ratio * min(previous_size, size)))
                start = supports[-1][-1] + 1 - overlap

            end = start + size

            if end > total_size:
                raise ValueError(
                    "The requested diagonal supports do not fit. "
                    "Reduce n_clusters, size_fraction, overlap_ratio, or size_jitter."
                )

            idx = np.arange(start, end)
            supports.append(idx)
            indicators[idx, k] = 1

    elif placement == "random":
        used = set()

        for k in range(n_clusters):
            size = sizes[k]

            if k == 0:
                idx = rng.choice(total_size, size=size, replace=False)

            else:
                previous = supports[-1]
                overlap = int(round(overlap_ratio * min(len(previous), size)))

                overlap_idx = (
                    rng.choice(previous, size=overlap, replace=False)
                    if overlap > 0
                    else np.array([], dtype=int)
                )

                new_needed = size - overlap
                forbidden = set(overlap_idx.tolist())

                # Prefer new indices not already used, so overlap is mainly controlled.
                candidates = np.array(
                    [i for i in range(total_size) if i not in used and i not in forbidden]
                )

                if len(candidates) < new_needed:
                    # Fall back to any indices not already in the current overlap set.
                    candidates = np.array(
                        [i for i in range(total_size) if i not in forbidden]
                    )

                if len(candidates) < new_needed:
                    raise ValueError("Not enough indices available to create support.")

                new_idx = rng.choice(candidates, size=new_needed, replace=False)
                idx = np.concatenate([overlap_idx, new_idx])

            idx = np.sort(idx)
            supports.append(idx)
            indicators[idx, k] = 1
            used.update(idx.tolist())

    else:
        raise ValueError("placement must be either 'diagonal' or 'random'.")

    return indicators, supports

def _shuffle_outputs(data, row_indicators, col_indicators, rng):
    """
    Shuffle rows and columns and return shuffled data and shuffled truth masks.
    """
    row_perm = rng.permutation(data.shape[0])
    col_perm = rng.permutation(data.shape[1])

    shuffled_data = data[row_perm][:, col_perm]
    shuffled_row_indicators = row_indicators[row_perm, :]
    shuffled_col_indicators = col_indicators[col_perm, :]

    return (
        shuffled_data,
        shuffled_row_indicators,
        shuffled_col_indicators,
        row_perm,
        col_perm,
    )

def float_to_name(x):
    """
    Convert float to filename-safe string.
    Example: 0.75 -> '0p75'
    """
    return f"{x:.2f}".replace(".", "p")

def diagonal_supports_fit(fraction, n_clusters, overlap_ratio):
    """
    Checks approximately whether diagonal supports of equal size can fit.

    Required fraction is:
        fraction * (n_clusters - (n_clusters - 1) * overlap_ratio)

    This must be <= 1.
    """
    required_fraction = fraction * (n_clusters - (n_clusters - 1) * overlap_ratio)
    return required_fraction <= 1.0