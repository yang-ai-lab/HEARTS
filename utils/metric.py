from typing import Dict, Sequence, Tuple

import numpy as np
from loguru import logger


class Metrics:
    """
    Class for calculating various metrics.
    """

    @staticmethod
    def iou(pred, gt) -> float:
        """
        Compute Intersection over Union (IoU) for ranges.

        Args:
            pred: Tuple of (start, end) or list of such tuples for predicted range(s).
            gt: Tuple of (start, end) or list of such tuples for ground truth range(s).

        Returns:
            IoU value as a float between 0 and 1.
        """

        def to_ranges(x):
            if (
                isinstance(x, (list, tuple))
                and len(x) == 2
                and all(isinstance(v, (int, float, type(None))) for v in x)
            ):
                return [x]
            elif isinstance(x, (list, tuple)):
                return list(x)
            else:
                raise ValueError(
                    "Input must be a tuple (start, end) or list of such tuples."
                )

        pred_ranges = to_ranges(pred)
        gt_ranges = to_ranges(gt)

        # Remove invalid ranges
        pred_ranges = [
            r
            for r in pred_ranges
            if r[0] is not None and r[1] is not None and r[0] < r[1]
        ]
        gt_ranges = [
            r
            for r in gt_ranges
            if r[0] is not None and r[1] is not None and r[0] < r[1]
        ]
        if not pred_ranges or not gt_ranges:
            return 0.0

        # Compute intersection
        inter = 0.0
        for pr in pred_ranges:
            for gr in gt_ranges:
                inter_start = max(pr[0], gr[0])
                inter_end = min(pr[1], gr[1])
                if inter_start < inter_end:
                    inter += inter_end - inter_start

        # Compute union
        def merge_ranges(ranges):
            if not ranges:
                return []
            sorted_ranges = sorted(ranges, key=lambda x: x[0])
            merged = [sorted_ranges[0]]
            for current in sorted_ranges[1:]:
                last = merged[-1]
                if current[0] <= last[1]:
                    merged[-1] = (last[0], max(last[1], current[1]))
                else:
                    merged.append(current)
            return merged

        merged_pred = merge_ranges(pred_ranges)
        merged_gt = merge_ranges(gt_ranges)
        union = (
            sum(r[1] - r[0] for r in merged_pred)
            + sum(r[1] - r[0] for r in merged_gt)
            - inter
        )
        if union <= 0:
            return 0.0
        return inter / union

    @staticmethod
    def mae(preds: Dict[str, Sequence], gts: Dict[str, Sequence]) -> Dict[str, float]:
        """
        Calculates the Mean Absolute Error (MAE) between predicted and ground truth sequences for each key.
        Args:
            preds (Dict[str, Sequence]): Dictionary of predicted sequences, keyed by identifier.
            gts (Dict[str, Sequence]): Dictionary of ground truth sequences, keyed by identifier.
        Returns:
            Dict[str, float]: Dictionary mapping each key to its MAE value.
        """
        out = {}
        for key in gts.keys():
            error = np.abs(np.array(preds[key]) - np.array(gts[key]))
            mae = np.mean(error)
            out[key] = float(mae)
        return out

    @staticmethod
    def mse(preds: Dict[str, Sequence], gts: Dict[str, Sequence]) -> Dict[str, float]:
        """
        Calculates the Mean Squared Error (MSE) between predicted and ground truth sequences for each key.
        Args:
            preds (Dict[str, Sequence]): Dictionary of predicted sequences, keyed by identifier.
            gts (Dict[str, Sequence]): Dictionary of ground truth sequences, keyed by identifier.
        Returns:
            Dict[str, float]: Dictionary mapping each key to its MSE value.
        """
        out = {}
        for key in gts.keys():
            error = np.array(preds[key]) - np.array(gts[key])
            mse = np.mean(error**2)
            out[key] = float(mse)
        return out

    @staticmethod
    def mape(preds: Dict[str, Sequence], gts: Dict[str, Sequence]) -> Dict[str, float]:
        """
        Calculates the Mean Absolute Percentage Error (MAPE) between predicted and ground truth sequences for each key.
        Args:
            preds (Dict[str, Sequence]): Dictionary of predicted sequences, keyed by identifier.
            gts (Dict[str, Sequence]): Dictionary of ground truth sequences, keyed by identifier.
        Returns:
            Dict[str, float]: Dictionary mapping each key to its MAPE value.
        """
        out = {}
        for key in gts.keys():
            if key not in preds:
                logger.error(f"Key {key} not found in predictions")
                continue

            try:
                pred_array = np.array(preds[key], dtype=float)
                gt_array = np.array(gts[key], dtype=float)
            except (ValueError, TypeError) as e:
                logger.error(
                    f"Could not convert sequences to numeric arrays for key {key}: {e}"
                )
                continue

            # Handle scalar inputs (0-dimensional arrays) by treating them as single-element arrays
            is_scalar_pred = pred_array.ndim == 0
            is_scalar_gt = gt_array.ndim == 0

            if is_scalar_pred and is_scalar_gt:
                # Both are scalars - treat as single-element arrays
                pred_len = 1
                gt_len = 1
            elif is_scalar_pred:
                # Only prediction is scalar
                pred_len = 1
                gt_len = len(gt_array) if gt_array.size > 0 else 0
            elif is_scalar_gt:
                # Only ground truth is scalar
                pred_len = len(pred_array) if pred_array.size > 0 else 0
                gt_len = 1
            else:
                # Both are arrays
                pred_len = len(pred_array) if pred_array.size > 0 else 0
                gt_len = len(gt_array) if gt_array.size > 0 else 0

            # Check for empty sequences
            if pred_len == 0 or gt_len == 0:
                logger.error(
                    f"Empty sequence for key {key}: pred_len={pred_len}, gt_len={gt_len}"
                )
                continue

            # Check for length mismatch before any masking
            if pred_len != gt_len:
                logger.error(
                    f"Sequence length mismatch for key {key}: pred={pred_len}, gt={gt_len}"
                )
                continue

            # Avoid division by zero
            nonzero_mask = gt_array != 0

            # Check if all ground truth values are zero
            if gt_len > 0 and np.all(~nonzero_mask):
                logger.error(
                    f"All GT values are zero for key {key}, MAPE is set to infinity."
                )
                out[key] = float("inf")  # All ground truth values are zero
                continue

            invalid_gt_count = np.sum(~nonzero_mask)
            if invalid_gt_count > 0:
                logger.warning(
                    f"Ground truth values for key {key} contain {invalid_gt_count} zeros, MAPE is calculated excluding these values."
                )

            # Apply masking to exclude zero ground truth values
            masked_pred_array = pred_array[nonzero_mask]
            masked_gt_array = gt_array[nonzero_mask]

            # If all values were masked out, skip this key
            if len(masked_pred_array) == 0:
                logger.error(
                    f"All GT values are zero for key {key}, skipping MAPE calculation"
                )
                continue

            error = np.abs((masked_pred_array - masked_gt_array) / masked_gt_array)
            mape = np.mean(error) * 100

            # Ensure NaN/inf values are preserved
            if not np.isfinite(mape):
                out[key] = float(mape)  # Keep NaN/inf as is
            else:
                out[key] = float(mape)

        return out

    @staticmethod
    def acc(preds: list, gts: list) -> Tuple[float, int]:
        """
        Calculates the accuracy between predicted and ground truth labels.
        Args:
            preds (list): List of predicted labels, failure should be represented as None.
            gts (list): List of ground truth labels.
        Returns:
            float: Accuracy value as a float between 0 and 1.
            int: failure count
        """
        if len(gts) == 0:
            return 0.0, 0
        failure_count = 0
        correct_count = 0

        for pred, gt in zip(preds, gts):
            if pred is None:
                failure_count += 1
            elif pred == gt:
                correct_count += 1

        total_count = len(gts) - failure_count
        if total_count == 0:
            return 0.0, failure_count
        accuracy = correct_count / total_count
        return accuracy, failure_count

    @staticmethod
    def smape(
        preds: Dict[str, Sequence], gts: Dict[str, Sequence]
    ) -> Tuple[Dict[str, float], int]:
        """
        Calculates the SMAPE-based accuracy score between predicted and ground truth sequences for each key.

        This function computes a bounded SMAPE (0-100% error) and converts it to an accuracy metric
        where 1.0 indicates perfect predictions and 0.0 indicates the worst possible predictions.

        Args:
            preds (Dict[str, Sequence]): Dictionary of predicted sequences, keyed by identifier.
            gts (Dict[str, Sequence]): Dictionary of ground truth sequences, keyed by identifier.

        Returns:
            Tuple[Dict[str, float], int]: A tuple containing:
                - Dict[str, float]: Dictionary mapping each key to its accuracy score (0.0-1.0).
                - int: count of invalid ground truth values (zeros)
        """
        out = {}
        total_invalid_gt_count = 0

        for key in preds.keys():
            if key not in gts:
                logger.error(f"Key {key} not found in ground truth")
                continue

            try:
                pred_array = np.array(preds[key], dtype=float)
                gt_array = np.array(gts[key], dtype=float)
            except (ValueError, TypeError) as e:
                logger.error(
                    f"Could not convert sequences to numeric arrays for key {key}: {e}"
                )
                continue

            # Handle scalar inputs (0-dimensional arrays) by treating them as single-element arrays
            is_scalar_pred = pred_array.ndim == 0
            is_scalar_gt = gt_array.ndim == 0

            if is_scalar_pred and is_scalar_gt:
                # Both are scalars - treat as single-element arrays
                pred_len = 1
                gt_len = 1
            elif is_scalar_pred:
                # Only prediction is scalar
                pred_len = 1
                gt_len = len(gt_array) if gt_array.size > 0 else 0
            elif is_scalar_gt:
                # Only ground truth is scalar
                pred_len = len(pred_array) if pred_array.size > 0 else 0
                gt_len = 1
            else:
                # Both are arrays
                pred_len = len(pred_array) if pred_array.size > 0 else 0
                gt_len = len(gt_array) if gt_array.size > 0 else 0

            # Check for empty sequences
            if pred_len == 0 or gt_len == 0:
                logger.error(
                    f"Empty sequence for key {key}: pred_len={pred_len}, gt_len={gt_len}"
                )
                continue

            # Check for length mismatch before any masking
            if pred_len != gt_len:
                logger.error(
                    f"Sequence length mismatch for key {key}: pred={pred_len}, gt={gt_len}"
                )
                continue

            gt_zero_mask = gt_array == 0

            # Check if all ground truth values are zero (but not if array is empty)
            if gt_len > 0 and np.all(gt_zero_mask):
                total_invalid_gt_count += gt_len
                logger.error(
                    f"All GT values are zero for key {key}, cannot compute SMAPE."
                )
                continue

            invalid_gt_count = np.sum(gt_zero_mask)
            total_invalid_gt_count += invalid_gt_count

            # Apply masking only after validation
            masked_pred_array = pred_array[~gt_zero_mask]
            masked_gt_array = gt_array[~gt_zero_mask]

            # If all values were masked out, skip this key
            if len(masked_pred_array) == 0:
                logger.error(
                    f"All GT values are zero for key {key}, skipping SMAPE calculation"
                )
                continue

            # Bounded SMAPE = (100/n) * Σ(|y_true - y_pred| / (|y_true| + |y_pred|))
            numerator = np.abs(masked_gt_array - masked_pred_array)
            denominator = np.abs(masked_gt_array) + np.abs(masked_pred_array)

            # Use masked array to handle zero denominators automatically
            denominator_masked = np.ma.masked_equal(denominator, 0)
            error = numerator / denominator_masked

            # Calculate bounded SMAPE, masked values contribute 0 to the mean
            bounded_smape = np.ma.mean(error) * 100

            # Convert to accuracy metric: 1.0 = perfect, 0.0 = worst
            smape = (100.0 - bounded_smape) / 100.0

            # Ensure NaN/inf values are preserved
            if not np.isfinite(smape):
                out[key] = float(smape)  # Keep NaN/inf as is
            else:
                out[key] = float(smape)

        return out, int(total_invalid_gt_count)
