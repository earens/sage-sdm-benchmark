"""Area Under the Precision-Recall-Gain curve (AUPRG), Flach & Kull (NeurIPS 2015).

Provides ``MultilabelAUPRG``, a ``torchmetrics.Metric`` computing per-label (or
macro/weighted) AUPRG for multilabel classification, with support for an
``ignore_index`` so out-of-range species can be excluded from the score.
"""

import torch
from torch import Tensor
from torchmetrics import Metric
import numpy as np
import os



def _precision_gain(tp: np.ndarray, fn: np.ndarray, fp: np.ndarray, tn: np.ndarray) -> np.ndarray:
    """Calculate Precision Gain, matching the reference implementation."""
    n_pos = tp + fn
    n_neg = fp + tn
    with np.errstate(divide='ignore', invalid='ignore'):
        pg = 1.0 - (n_pos / n_neg) * (fp / tp)
    if np.isscalar(pg):
        if tn + fn == 0:
            pg = 0.0
    else:
        pg[tn + fn == 0] = 0.0
    return pg


def _recall_gain(tp: np.ndarray, fn: np.ndarray, fp: np.ndarray, tn: np.ndarray) -> np.ndarray:
    """Calculate Recall Gain, matching the reference implementation."""
    n_pos = tp + fn
    n_neg = fp + tn
    with np.errstate(divide='ignore', invalid='ignore'):
        rg = 1.0 - (n_pos / n_neg) * (fn / tp)
    if np.isscalar(rg):
        if tn + fn == 0:
            rg = 1.0
    else:
        rg[tn + fn == 0] = 1.0
    return rg


def _create_segments(labels: np.ndarray, pos_scores: np.ndarray, neg_scores: np.ndarray):
    """Group consecutive samples with the same scores into segments."""
    n = len(labels)
    # Sort by decreasing pos_scores, breaking ties by increasing neg_scores
    new_order = np.lexsort((neg_scores, -pos_scores))
    labels = labels[new_order]
    pos_scores = pos_scores[new_order]
    neg_scores = neg_scores[new_order]

    seg_pos_score = np.zeros(n)
    seg_neg_score = np.zeros(n)
    seg_pos_count = np.zeros(n)
    seg_neg_count = np.zeros(n)
    j = -1
    for i in range(n):
        if (i == 0) or (pos_scores[i - 1] != pos_scores[i]) or (neg_scores[i - 1] != neg_scores[i]):
            j += 1
            seg_pos_score[j] = pos_scores[i]
            seg_neg_score[j] = neg_scores[i]
        if labels[i] == 0:
            seg_neg_count[j] += 1
        else:
            seg_pos_count[j] += 1

    return (seg_pos_score[:j + 1], seg_neg_score[:j + 1],
            seg_pos_count[:j + 1], seg_neg_count[:j + 1])


def _create_crossing_points(
    pos_score: np.ndarray, neg_score: np.ndarray,
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, tn: np.ndarray,
    rec_gain: np.ndarray, prec_gain: np.ndarray,
    n_pos: float, n_neg: float
):
    """
    Create crossing points matching the reference implementation.

    Introduces interpolated points where:
      1. The curve crosses the y-axis (recall_gain = 0)
      2. The curve crosses precision_gain = 0 in the non-negative recall_gain region
    """
    n = n_pos + n_neg
    is_crossing = np.zeros(len(pos_score))

    def _insert_point(idx, new_ps, new_ns, new_tp, new_fp, new_fn, new_tn, new_pg, new_rg, new_cr):
        """Insert a new point and re-sort all arrays by recall_gain. Returns updated arrays."""
        nonlocal pos_score, neg_score, tp, fp, fn, tn, prec_gain, rec_gain, is_crossing
        pos_score = np.insert(pos_score, 0, new_ps)
        neg_score = np.insert(neg_score, 0, new_ns)
        tp = np.insert(tp, 0, new_tp)
        fp = np.insert(fp, 0, new_fp)
        fn = np.insert(fn, 0, new_fn)
        tn = np.insert(tn, 0, new_tn)
        prec_gain = np.insert(prec_gain, 0, new_pg)
        rec_gain = np.insert(rec_gain, 0, new_rg)
        is_crossing = np.insert(is_crossing, 0, new_cr)
        order = np.lexsort((-prec_gain, rec_gain))
        pos_score = pos_score[order]
        neg_score = neg_score[order]
        tp = tp[order]
        fp = fp[order]
        fn = fn[order]
        tn = tn[order]
        prec_gain = prec_gain[order]
        rec_gain = rec_gain[order]
        is_crossing = is_crossing[order]

    # 1. Crossing through the y-axis (recall_gain = 0)
    j = int(np.amin(np.where(rec_gain >= 0)[0]))
    if rec_gain[j] > 0:
        # Interpolate in the contingency table space
        keys = ['pos_score', 'neg_score', 'TP', 'FP', 'FN', 'TN', 'prec_gain', 'rec_gain']
        arrs = [pos_score, neg_score, tp, fp, fn, tn, prec_gain, rec_gain]
        point_1 = np.array([a[j] for a in arrs])
        point_2 = np.array([a[j - 1] for a in arrs])
        delta = point_1 - point_2
        tp_idx = keys.index('TP')
        if delta[tp_idx] > 0:
            alpha = (n_pos * n_pos / n - tp[j - 1]) / delta[tp_idx]
        else:
            alpha = 0.5

        with np.errstate(invalid='ignore'):
            new_point = point_2 + alpha * delta

        new_pg = _precision_gain(new_point[tp_idx], new_point[keys.index('FN')],
                                 new_point[keys.index('FP')], new_point[keys.index('TN')])
        _insert_point(0, new_point[0], new_point[1],
                       new_point[tp_idx], new_point[keys.index('FP')],
                       new_point[keys.index('FN')], new_point[keys.index('TN')],
                       float(new_pg), 0.0, 1.0)

    # 2. Crossings through precision_gain = 0 in non-negative recall_gain region
    x = rec_gain
    y = prec_gain
    temp_y_0 = np.append(y, 0)
    temp_0_y = np.append(0, y)
    temp_1_x = np.append(1, x)
    with np.errstate(invalid='ignore'):
        indices = np.where(np.logical_and((temp_y_0 * temp_0_y < 0), (temp_1_x >= 0)))[0]

    for i in indices:
        cross_x = x[i - 1] + (-y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1])

        keys = ['pos_score', 'neg_score', 'TP', 'FP', 'FN', 'TN', 'prec_gain', 'rec_gain']
        arrs = [pos_score, neg_score, tp, fp, fn, tn, prec_gain, rec_gain]
        point_1 = np.array([a[i] for a in arrs])
        point_2 = np.array([a[i - 1] for a in arrs])
        delta = point_1 - point_2
        tp_idx = keys.index('TP')
        fp_idx = keys.index('FP')
        if delta[tp_idx] > 0:
            alpha = (n_pos * n_pos / (n - n_neg * cross_x) - tp[i - 1]) / delta[tp_idx]
        else:
            alpha = (n_neg / n_pos * tp[i - 1] - fp[i - 1]) / delta[fp_idx]

        with np.errstate(invalid='ignore'):
            new_point = point_2 + alpha * delta

        new_rg = _recall_gain(new_point[tp_idx], new_point[keys.index('FN')],
                              new_point[keys.index('FP')], new_point[keys.index('TN')])
        _insert_point(0, new_point[0], new_point[1],
                       new_point[tp_idx], new_point[keys.index('FP')],
                       new_point[keys.index('FN')], new_point[keys.index('TN')],
                       0.0, float(new_rg), 1.0)
        # Re-read updated arrays after insertion
        indices += 1
        x = rec_gain
        y = prec_gain
        temp_y_0 = np.append(y, 0)
        temp_0_y = np.append(0, y)
        temp_1_x = np.append(1, x)

    return pos_score, neg_score, tp, fp, fn, tn, rec_gain, prec_gain, is_crossing


def _compute_auprg_single(preds: np.ndarray, target: np.ndarray) -> float:
    """
    Compute AUPRG for a single binary classification problem.

    Matches the reference implementation from Flach & Kull (2015):
        1. Sort samples by descending prediction score.
        2. Compute cumulative TP/FP at each threshold (one per sample group).
        3. Convert to PRG coordinates.
        4. Create crossing points where the curve crosses recall_gain=0
           and precision_gain=0.
        5. Integrate using trapezoidal rule over the region where recall_gain >= 0.

    Args:
        preds: Prediction scores, shape (n_samples,).
        target: Binary targets (0 or 1), shape (n_samples,).

    Returns:
        AUPRG score, or NaN if undefined.
    """
    n_pos = float(target.sum())
    n_neg = float(len(target) - n_pos)

    if n_pos == 0 or n_neg == 0 or len(target) == 0:
        return float('nan')

    # Convert labels: ensure positives are 1, rest are 0
    labels = (target == 1).astype(np.float64)
    pos_scores = preds.astype(np.float64)
    neg_scores = -pos_scores

    # Create segments (groups of samples with the same scores)
    seg_pos_score, seg_neg_score, seg_pos_count, seg_neg_count = \
        _create_segments(labels, pos_scores, neg_scores)

    # Build the operating points table (one entry per threshold + origin)
    ps = np.insert(seg_pos_score, 0, np.inf)
    ns = np.insert(seg_neg_score, 0, -np.inf)
    tp = np.insert(np.cumsum(seg_pos_count), 0, 0.0)
    fp = np.insert(np.cumsum(seg_neg_count), 0, 0.0)
    fn = n_pos - tp
    tn = n_neg - fp

    pg = _precision_gain(tp, fn, fp, tn)
    rg = _recall_gain(tp, fn, fp, tn)

    # Create crossing points (matching reference implementation)
    ps, ns, tp, fp, fn, tn, rg, pg, is_crossing = \
        _create_crossing_points(ps, ns, tp, fp, fn, tn, rg, pg, n_pos, n_neg)

    # Integrate: trapezoidal rule over consecutive points where recall_gain >= 0
    # This matches calc_auprg from the reference implementation exactly
    area = 0.0
    for i in range(1, len(rg)):
        if (not np.isnan(rg[i - 1])) and (rg[i - 1] >= 0):
            width = rg[i] - rg[i - 1]
            height = (pg[i] + pg[i - 1]) / 2.0
            area += width * height

    return float(area)


def _compute_auprg_worker(
    label_idx: int,
    preds_valid: np.ndarray,
    target_valid: np.ndarray,
) -> tuple[int, float]:
    """Top-level worker for process-pool-based AUPRG computation."""
    return label_idx, _compute_auprg_single(preds_valid, target_valid)


class MultilabelAUPRG(Metric):
    """
    Area Under the Precision-Recall-Gain Curve for multilabel classification.

    Implements the PRG framework from Flach & Kull (2015):

        PrecisionGain = 1 - (π / (1-π)) * (FP / TP)
        RecallGain    = 1 - (π / (1-π)) * (FN / TP)

    where π = n_pos / (n_pos + n_neg) is the proportion of positives.

    Key implementation details matching the reference:
        - PRG coordinates from cumulative TP/FP at each score threshold
        - Interpolated crossing points at RecallGain=0 and PrecisionGain=0
        - Trapezoidal integration over the region where RecallGain >= 0

    Args:
        num_labels: Number of labels in multilabel classification.
        average: Averaging method ('macro', 'weighted', or None for per-label scores).
        ignore_index: Target value to ignore (e.g., for out-of-range species).
    Example:
        >>> metric = MultilabelAUPRG(num_labels=10, average='macro')
        >>> preds = torch.rand(100, 10)
        >>> target = torch.randint(0, 2, (100, 10))
        >>> metric.update(preds, target)
        >>> auprg = metric.compute()

    Reference:
        Flach, P., & Kull, M. (2015). Precision-Recall-Gain Curves: PR Analysis Done Right.
        https://proceedings.neurips.cc/paper/2015/hash/33e8075e9970de0cfea955afd4644bb2-Abstract.html
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        num_labels: int,
        average: str | None = "macro",
        ignore_index: int | None = None,
    ):
        super().__init__()

        if average not in ("macro", "median", "weighted", None):
            raise ValueError(f"average must be 'macro', 'median', 'weighted', or None, got {average}")
        if ignore_index is not None and ignore_index in (0, 1):
            raise ValueError(f"ignore_index={ignore_index} conflicts with binary target values")

        self.num_labels = num_labels
        self.average = average
        self.ignore_index = ignore_index

        # Store predictions and targets on CPU to avoid GPU OOM
        self.preds_list: list[Tensor] = []
        self.target_list: list[Tensor] = []

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Store predictions and targets for later computation.

        Args:
            preds: Prediction probabilities of shape (batch_size, num_labels).
            target: Binary targets of shape (batch_size, num_labels).
        """
        if preds.shape != target.shape:
            raise ValueError(f"preds shape {preds.shape} != target shape {target.shape}")

        if self.ignore_index is not None:
            mask = target != self.ignore_index
            preds = preds.clone()
            preds[~mask] = float('nan')
            target = target.clone()
            target[~mask] = -1

        self.preds_list.append(preds.detach().cpu())
        self.target_list.append(target.detach().cpu())

    def compute(self) -> Tensor:
        """
        Compute AUPRG from all stored predictions and targets.

        Uses ProcessPoolExecutor to parallelise the per-label computation.
        Labels are submitted in small batches to avoid building a massive
        work list that peaks at ~1 GB for 5771 labels.

        Returns:
            AUPRG score. Shape depends on `average` parameter:
            - 'macro' or 'weighted': scalar
            - None: tensor of shape (num_labels,)
        """
        if len(self.preds_list) == 0:
            if self.average is None:
                return torch.full((self.num_labels,), float('nan'))
            return torch.tensor(float('nan'))

        # Pre-allocate and copy to avoid torch.cat creating a temporary
        # tensor that doubles peak memory.
        n_total = sum(t.shape[0] for t in self.preds_list)
        all_preds = np.empty((n_total, self.num_labels), dtype=np.float32)
        all_target = np.empty((n_total, self.num_labels), dtype=np.float32)
        offset = 0
        for p, t in zip(self.preds_list, self.target_list):
            n = p.shape[0]
            all_preds[offset:offset + n] = p.numpy()
            all_target[offset:offset + n] = t.numpy()
            offset += n

        # Free the accumulated lists early — all_preds/all_target hold the data
        self.preds_list.clear()
        self.target_list.clear()

        auprg_scores = np.full(self.num_labels, np.nan, dtype=np.float32)

        n_workers = max(1, min(int(os.environ.get("AUPRG_WORKERS",
                                                   min(os.cpu_count() or 1, 8))),
                               self.num_labels))

        # Process labels in batches to limit peak memory.
        # Each batch extracts per-label arrays, submits to the pool,
        # collects results, then the batch arrays are freed.
        BATCH_SIZE = 256

        # The per-label AUPRG work is pure-Python (GIL-bound), so a thread pool
        # runs on a single core. A process pool gives real parallelism across the
        # thousands of species (same fork model the DataLoader workers use).
        # Set AUPRG_WORKERS=1 to force serial execution.
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for batch_start in range(0, self.num_labels, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, self.num_labels)
                futures = []
                for label_idx in range(batch_start, batch_end):
                    preds_label = all_preds[:, label_idx]
                    target_label = all_target[:, label_idx]
                    valid = (target_label >= 0) & np.isfinite(preds_label)
                    if valid.sum() == 0:
                        continue
                    futures.append(pool.submit(
                        _compute_auprg_worker, label_idx,
                        preds_label[valid],
                        target_label[valid].astype(np.float64),
                    ))
                for f in futures:
                    idx, val = f.result()
                    auprg_scores[idx] = val

        auprg_tensor = torch.from_numpy(auprg_scores)

        if self.average is None:
            return auprg_tensor

        # Only average over valid (non-NaN) labels
        valid_mask = torch.isfinite(auprg_tensor)

        if valid_mask.sum() == 0:
            return torch.tensor(float('nan'))

        if self.average == "median":
            return torch.tensor(float(np.nanmedian(auprg_scores)))

        if self.average == "macro":
            return auprg_tensor[valid_mask].mean()

        elif self.average == "weighted":
            pos_counts = torch.tensor([
                float(((all_target[:, i] == 1) & (all_target[:, i] != -1)).sum())
                for i in range(self.num_labels)
            ])
            weights = pos_counts / (pos_counts.sum() + 1e-10)
            weights = weights * valid_mask.float()

            if weights.sum() == 0:
                return torch.tensor(float('nan'))

            return (auprg_tensor.nan_to_num(0) * weights).sum() / weights.sum()

        return auprg_tensor

    def reset(self) -> None:
        """Reset metric state."""
        super().reset()
        self.preds_list = []
        self.target_list = []
