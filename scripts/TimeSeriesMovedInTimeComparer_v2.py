import torch
import torch.nn.functional as F
from enum import Enum
import time

class TimeSeriesMovedInTimeComparer:
    class ComparisonMode(Enum):
        DOMINANT = "Dominant"
        MEAN = "Mean"

    def __init__(self, mode: ComparisonMode, similarity_threshold: float = 0.1, size_of_kernel: int = 30, bypass: tuple = (0, 0), logger=None):
        self.comparisonMode = mode
        self.similarity_threshold = similarity_threshold
        self.size_of_kernel = size_of_kernel
        self.bypass = bypass
        self.logger = logger

    def _normalize_v2(self, input_series: torch.Tensor) -> torch.Tensor:
        """
        Normalizes input series by subtracting channel-specific means and the mean of the first channel.

        Args:
            input_series (torch.Tensor): Shape [batch_size, channels, length]

        Returns:
            torch.Tensor: Normalized tensor [batch_size, channels, length]
        """
        mean_values = torch.tensor([0.0, -0.32172874023188663, 0.9329161398211201, 1.050562329499409],
                                   device=input_series.device, dtype=torch.float32)
        channel_mean = input_series[:, 0, :].mean(dim=-1)  # [batch_size]
        return input_series - mean_values[None, :, None] - channel_mean[:, None, None]

    def _change_series_to_kernels(self, series: torch.Tensor, size: int, bypass: tuple = (0, 0)) -> torch.Tensor:
        """
        Extracts kernels from the series.

        Args:
            series (torch.Tensor): Shape [batch_size, channels, length]
            size (int): Kernel size
            bypass (tuple): (left_bypass, right_bypass)

        Returns:
            torch.Tensor: Shape [batch_size, channels, num_kernels, size]
        """
        left_bypass, right_bypass = bypass
        analyzed_series = series[:, :, left_bypass: -right_bypass if right_bypass > 0 else None]
        unfolded = analyzed_series.unfold(2, size, size)  # [batch_size, channels, num_kernels, size]
        return unfolded

    def _compute_distances_and_counters(self, similar_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes distances and counters for similar points in each kernel.

        Args:
            similar_points (torch.Tensor): Shape [num_kernels, length]

        Returns:
            tuple: (distances [num_kernels], counters [num_kernels])
        """
        num_kernels, length = similar_points.shape
        distances = torch.zeros(num_kernels, device=similar_points.device)
        counters = torch.zeros(num_kernels, dtype=torch.int32, device=similar_points.device)

        for k in range(num_kernels):
            indices = torch.nonzero(similar_points[k], as_tuple=False).flatten()
            if indices.numel() < 2:
                continue

            # Filter indices with minimum distance
            diffs = torch.diff(indices)
            mask = diffs >= self.size_of_kernel
            filtered_indices = torch.cat([indices[:1], indices[1:][mask]])

            if filtered_indices.numel() < 2:
                continue

            # Compute distances
            dists = torch.diff(filtered_indices).float()
            counters[k] = dists.numel()
            if self.comparisonMode == self.ComparisonMode.MEAN:
                distances[k] = dists.mean()
            else:  # DOMINANT
                rounded = dists.round(decimals=1)
                vals, counts = torch.unique(rounded, return_counts=True)
                distances[k] = vals[torch.argmax(counts)]

        return distances, counters

    def _calculate_mark_based_on_similar_repetition_distance_v2(self, distances: torch.Tensor, ref_channel: int, length: int) -> torch.Tensor:
        """
        Calculates mark based on distances of similar repetitions.

        Args:
            distances (torch.Tensor): Shape [ref_channels, comp_channels, num_kernels]
            ref_channel (int): Reference channel index
            length (int): Length of the series

        Returns:
            torch.Tensor: Scalar mark
        """
        base_distances = distances[ref_channel, 0, :]  # Compare to base channel
        mse = torch.tensor(0.0, device=distances.device)
        for comp_channel in range(distances.shape[1]):
            if comp_channel == 0:
                continue
            comp_distances = distances[ref_channel, comp_channel, :]
            valid_mask = (base_distances > 0) & (comp_distances > 0)
            if valid_mask.any():
                mse += ((base_distances[valid_mask] - comp_distances[valid_mask]) ** 2).mean()
        return mse / max(length, 1)

    def _calculate_mark_based_on_similar_repetition_counter(self, counters: torch.Tensor, ref_channel: int) -> torch.Tensor:
        """
        Calculates mark based on counters of similar repetitions.

        Args:
            counters (torch.Tensor): Shape [ref_channels, comp_channels, num_kernels]
            ref_channel (int): Reference channel index

        Returns:
            torch.Tensor: Scalar mark
        """
        base_counter = counters[ref_channel, 0, :]  # Compare to base channel
        mse = torch.tensor(0.0, device=counters.device)
        for comp_channel in range(counters.shape[1]):
            if comp_channel == 0:
                continue
            comp_counter = counters[ref_channel, comp_channel, :]
            valid_mask = (base_counter > 0) & (comp_counter > 0)
            if valid_mask.any():
                mse += ((base_counter[valid_mask] - comp_counter[valid_mask]) ** 2).mean()
        return mse

    def compare(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Compares time series data across channels and returns a score for each sample.

        Args:
            input_data (torch.Tensor): Shape [batch_size, channels, length]

        Returns:
            torch.Tensor: Shape [batch_size, 1]
        """
        assert len(input_data.shape) == 3, "Bad input shape of comparing data"
        batch_size, channels, length = input_data.shape
        device = input_data.device

        # Normalize
        normalized_sample = self._normalize_v2(input_data)

        # Extract kernels
        kernels = self._change_series_to_kernels(normalized_sample, self.size_of_kernel, self.bypass)
        num_kernels = kernels.shape[2]

        # Pad and unfold series
        padded_series = F.pad(normalized_sample, (self.size_of_kernel // 2, self.size_of_kernel // 2), "constant", 0)
        unfolded_series = padded_series.unfold(2, self.size_of_kernel, 1)

        # Compute MSE tensor
        kernels_expanded = kernels[:, :, :, None, :]  # [batch_size, ref_channels, num_kernels, 1, kernel_size]
        series_expanded = unfolded_series[:, None, :, :, :]  # [batch_size, 1, comp_channels, length, kernel_size]
        mse_tensor = (kernels_expanded - series_expanded).pow(2).mean(dim=-1)  # [batch_size, ref_channels, comp_channels, num_kernels, length]

        # Find similar points
        similar_points = mse_tensor < self.similarity_threshold

        # Compute distances and counters
        distance_tensor = torch.zeros(batch_size, channels, channels, num_kernels, device=device)
        counter_tensor = torch.zeros(batch_size, channels, channels, num_kernels, dtype=torch.int32, device=device)
        for b in range(batch_size):
            for ref in range(channels):
                for comp in range(channels):
                    sp = similar_points[b, ref, comp, :, :]
                    dist, cnt = self._compute_distances_and_counters(sp)
                    distance_tensor[b, ref, comp, :] = dist
                    counter_tensor[b, ref, comp, :] = cnt

        # Compute marks
        marks = torch.zeros(batch_size, device=device)
        for b in range(batch_size):
            for ref in range(channels):
                mark_dist = self._calculate_mark_based_on_similar_repetition_distance_v2(
                    distance_tensor[b], ref, length
                )
                mark_cnt = self._calculate_mark_based_on_similar_repetition_counter(
                    counter_tensor[b], ref
                )
                marks[b] += (mark_dist + mark_cnt) / channels

        return marks.unsqueeze(1)

if __name__ == "__main__":
    # Example usage
    batch_size, channels, length = 2, 4, 2200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_data = torch.randn(batch_size, channels, length).to(device)
    comparer = TimeSeriesMovedInTimeComparer(TimeSeriesMovedInTimeComparer.ComparisonMode.DOMINANT)
    start_time = time.time()
    result = comparer.compare(real_data)
    stop_time = time.time()
    print(f"Execution time: {stop_time - start_time} seconds")
    print(f"Result shape: {result.shape}")
    print(f"Result: {result}")