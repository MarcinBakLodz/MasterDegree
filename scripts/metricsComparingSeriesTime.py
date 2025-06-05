"""Moduł służy do oceny podobieństwa pomiędzy channelami w zbiorze odczytów z rurociągu paliwowego (time series data transitioned in time), poprzez:
    1) Okreslenie punktów charakterystycznych modelu (kernele).
    2) Określenie podobieństwa analizowanych punktów do kerneli.
    3) Policzenie liczbę punktów podobnych i odległości między nimi dla każdego kernela w próbce.
    4) Zwrócenie jednej wartości dla każdej próbki.  
    
    arguments:
        -input ([batch_size, sequence_lenght, channels])
    
    results:
        -grade ([batch_size, 1])
"""

from enum import Enum
from torch.utils.data import TensorDataset, DataLoader, random_split
from datasets import LocalizationDataFormat
from typing import List
from collections import Counter
from time import time
import torch.nn.functional as F
import torch
import data_preprocessing
import matplotlib.pyplot as plt
import logging



class TimeSeriesMovedInTimeComparer():
    class ComparisonMode(Enum):
        MEAN = 0,
        DOMINANT = 1
    
    def __init__(self, comparison_mode:ComparisonMode, size_of_kernel:int = 30, bypass:tuple = (0,0), similarity_threshold:float = 0.0125, logger = None):
        self.comparisonMode = comparison_mode
        self.size_of_kernel = size_of_kernel
        self.bypass = bypass
        self.similarity_threshold = similarity_threshold
        
        self._initialize_logger(logger)
        
    def _initialize_logger(self, logger):
        if logger == None:
            self.logger = logging.getLogger("TimeSeriesComparer")
            if not self.logger.hasHandlers():
                self.logger.setLevel(logging.DEBUG)
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.DEBUG)
                formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
                self.logger.propagate = False
        else:
            self.logger = logger
            

    def compare(self, input_data:torch.Tensor)->torch.Tensor:
        """Główna funkcja do porównywania

        Args:
            input_data (torch.Tensor):

        Returns:
            torch.Tensor: 
        """
        assert len(input_data.shape) == 3, "bad input shape of comparing data" # (batch, channels, lenght)
        #batch loop (4, 2200)
        batch_size = input_data.shape[0]
        for element_index in range(batch_size):
            sample = input_data[element_index]
            normalized_sample = self._normalize_v2(sample)
            self.logger.debug(f"normalized_sample:  {normalized_sample.shape}")
            
            channels_number = normalized_sample.shape[0]
            
            for refference_channel_index in range(channels_number):
                distance_between_similar_points_list = []
                similar_points_counter_list = []
                
                refference_channel_series = normalized_sample[refference_channel_index]
                self.logger.debug(f"refference_channe_series shape: {refference_channel_series.shape}")
                
                
                refference_kernels = self._change_one_series_to_kernels(refference_channel_series, self.size_of_kernel, self.bypass)
                self.logger.debug(f"refference_kernels: {refference_kernels.shape}")
                
                for comparison_channel_index in range(channels_number):
                    comparison_channel_series = normalized_sample[comparison_channel_index]
                    


                    similarity_value = self._apply_kerneled_mse_on_one_series(comparison_channel_series, refference_kernels)

                    similar_points = self._find_similar_regions(similarity_value, self.similarity_threshold)
                    
                    distance_between_similar_points = self._calculate_best_distance_between_similar_point(similar_points, self.comparisonMode)

                    # start_time_1 = time()
                    # stop_time_1 = time()
                    # assert torch.allclose(distance_between_similar_points, distance_between_similar_points_v2, rtol=1e-5, atol=1e-8), "Tensory nie są takie same"
                    # print(f"przed: {stop_time_1 - start_time_1}")
                    # print(f"po: {stop_time_2 - start_time_2}")

                    similar_points_counter = self.count_valid_similar_point_distances(similar_points)

                    distance_between_similar_points_list.append(distance_between_similar_points)
                    similar_points_counter_list.append(similar_points_counter)
                
                distance_between_similar_points_tensor = torch.stack(distance_between_similar_points_list)
                similar_points_counter_tensor = torch.stack(similar_points_counter_list)
                
                self.logger.debug(f"distance_between_similar_points_tensor: {distance_between_similar_points_tensor.shape}")
                self.logger.debug(f"similar_points_counter_tensor: {similar_points_counter_tensor.shape}")
                

                mark_similar_repetition_distances = self._calculate_mark_based_on_similar_repetition_distance_v2(distance_between_similar_points_tensor, refference_channel_index, comparison_channel_series.shape[0])
                # start_time_2 = time()
                # stop_time_2 = time()
                # assert torch.equal(mark_similar_repetition_distances, mark_similar_repetition_distances_v2), "Tensory nie są takie same"
                # print(f"przed: {stop_time_1 - start_time_1}")
                # print(f"po: {stop_time_2 - start_time_2}")

                mark_similar_repetition_counter = self._calculate_mark_based_on_similar_repetition_counter(similar_points_counter_tensor, refference_channel_index)
                self.logger.debug(f"mark_similar_repetition_counter: {mark_similar_repetition_counter} / {mark_similar_repetition_distances}")
        pass

    def _normalize(self, input_series:torch.Tensor)->torch.Tensor:
        """Przesunięcie danych do jednego poziomu
        
        Args:
            input_series (torch.Tensor) [channels, length]
            
        Returns:
            (torch.Tensor) [channels, length]
        """
        mean_values = [0.0, -0.32172874023188663, 0.9329161398211201, 1.050562329499409]
        result_list = []
        
        for channel_index in range(input_series.shape[0]):
            constant_shifting_y = input_series[channel_index] - mean_values[channel_index] - torch.mean(input_series[0])
            result_list.append(constant_shifting_y)
        result = torch.stack(result_list)
        return result
    
    def _normalize_v2(self, input_series: torch.Tensor) -> torch.Tensor:
        """
        Normalizes input series by subtracting channel-specific means and the mean of the first channel.

        Args:
            input_series (torch.Tensor): Shape [channels, length]

        Returns:
            torch.Tensor: Normalized tensor [channels, length]
        """
        mean_values = torch.tensor([0.0, -0.32172874023188663, 0.9329161398211201, 1.050562329499409],
                                 device=input_series.device, dtype=torch.float32)
        channel_mean = input_series[0].mean()
        return input_series - mean_values[:, None] - channel_mean
    
    def _change_one_series_to_kernels(self, series: torch.Tensor, size: int, bypass: tuple = (0, 0)) -> torch.Tensor:
        """
        Dzieli jednowymiarowy tensor na mniejsze fragmenty o zadanym rozmiarze (kernelach).

        Args:
            series (torch.Tensor): Jednowymiarowy tensor danych wejściowych.
            size (int): Rozmiar każdego kernela.
            bypass (tuple, optional): Liczba elementów do pominięcia na początku i końcu serii. Domyślnie (0, 0).

        Returns:
            torch.Tensor: Tensor o kształcie (N, 1, size), gdzie N to liczba wyodrębnionych kernelów.
        """
        self.logger.debug(f"\t\t\tchange_one_series_to_kernels - wejście: size={size}, bypass={bypass}")
        analized_series = series[bypass[0]:-bypass[1]] if sum(bypass) > 0 else series
        self.logger.debug(f"analized_series shape: {analized_series.shape}")
        number_of_kernels = analized_series.shape[-1] // size

        result = analized_series.unfold(0, size, size).unsqueeze(1)
        
        self.logger.debug(f"\t\t\tchange_one_series_to_kernels - liczba kernelów: {number_of_kernels}, kształt wyniku: {result.shape}")
        return result

    def _find_similar_regions(self, mse_result: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Zwraca tensor binarny z informacjami które obszary zostały zakwalifikowane jako podobne

        Args:
            mse_result (torch.Tensor): Tensor o kształcie (N, L), wynik działania _apply_kerneled_mse_on_one_series.
                                    N to liczba kernelów, L to długość serii.
            threshold (float): Wartość progowa, poniżej której traktujemy fragmenty jako podobne.

        Returns:
            torch.Tensor: Tensor typu bool o tym samym kształcie (N, L), gdzie True oznacza spełnienie warunku.
        """
        return mse_result < threshold

    def _calculate_best_distance_between_similar_point(self, input_series: torch.Tensor, comparison_mode: ComparisonMode) -> torch.Tensor:
        results: List[float] = []

        for channel in input_series:
            indices = torch.nonzero(channel == 1).flatten()

            # Filtrowanie indeksów: tylko punkty oddalone >=10 od poprzedniego wybranego
            filtered = []
            last_added = None
            for idx in indices:
                if last_added is None or (idx - last_added).item() >= self.size_of_kernel:
                    filtered.append(idx)
                    last_added = idx

            if len(filtered) < 2:
                results.append(0.0)
                continue

            filtered = torch.tensor(filtered, dtype=torch.int32)
            distances = (filtered[1:] - filtered[:-1]).float()

            if comparison_mode == self.ComparisonMode.MEAN:
                result = distances.mean().item()
            elif comparison_mode == self.ComparisonMode.DOMINANT:
                rounded = distances.round(decimals=1)
                vals, counts = torch.unique(rounded, return_counts=True)
                result = vals[torch.argmax(counts)].item()
            else:
                raise ValueError(f"Unsupported comparison mode: {comparison_mode}")

            results.append(result)

        return torch.tensor(results, dtype=torch.float32)
    
    def count_valid_similar_point_distances(self, input_series: torch.Tensor) -> torch.Tensor:
        counts: List[int] = []

        for channel in input_series:
            indices = torch.nonzero(channel == 1).flatten()

            # Filtrowanie indeksów: tylko punkty oddalone >=10 od poprzedniego wybranego
            filtered = []
            last_added = None
            for idx in indices:
                if last_added is None or (idx - last_added).item() >= self.size_of_kernel:
                    filtered.append(idx)
                    last_added = idx

            if len(filtered) < 2:
                counts.append(0)
            else:
                counts.append(len(filtered) - 1)

        return torch.tensor(counts, dtype=torch.int32)
    
    def _calculate_mark_based_on_similar_repetition_counter(self, input_data:torch.Tensor, base_channel_index:int):
        """Na podstawie tensora licznika powtórzeń danej cechy [channels, kernels]
        Dla siebie samego [kernels] oblicza dominantę powótrzeń w próbce [1], następnie mse dla każdej z cech
        Dla pozostałych bierze [channels,kernels] i liczy mse pomiędzy wszystkimi wartościami gdzie bazowym jest tensor  

        Args:
            input_data (torch.Tensor): [channels, kernels]
            base_channel_index (int):
        Returns:
            float: value based on repetition of all kernels
        """
        input_data = input_data.float()
        base = input_data[base_channel_index]
        result = 0.0
        num_channels = input_data.shape[0]

        for channel_index in range(num_channels):
            compare_tensor = input_data[channel_index]

            if channel_index == base_channel_index:
                nonzero_base = base[base > 1]
                if len(nonzero_base) == 0:
                    continue  # brak danych niezerowych

                counter = Counter(nonzero_base.tolist())
                dominant_value, _ = counter.most_common(1)[0]
                dominant_value = float(dominant_value)

                mse_input = (nonzero_base - dominant_value) ** 2
            else:
                # Pomijamy tylko te pozycje, gdzie obie są zerami
                mask = ~((base <= 1) & (compare_tensor <= 1))
                if not torch.any(mask):
                    continue  # brak wspólnych niezerowych danych

                mse_input = (base[mask] - compare_tensor[mask]) ** 2

            mse = torch.mean(mse_input)
            result += mse / num_channels

        return result 


    def _calculate_mark_based_on_similar_repetition_distance(self, input_data:torch.Tensor, base_channel_index:int, analized_data_lenght:int):
        """Na podstawie tensora licznika powtórzeń danej cechy [channels, kernels]
        Dla siebie samego [kernels] oblicza dominantę powótrzeń w próbce [1], następnie mse dla każdej z cech
        Dla pozostałych bierze [channels,kernels] i liczy mse pomiędzy wszystkimi wartościami gdzie bazowym jest tensor  

        Args:
            input_data (torch.Tensor): [channels, kernels]
            base_channel_index (int):
        Returns:
            float: value based on repetition of all kernels
        """
        input_data = input_data.float()
        base = input_data[base_channel_index]
        result = 0.0
        num_channels = input_data.shape[0]

        for channel_index in range(num_channels):
            compare_tensor = input_data[channel_index]

            if channel_index == base_channel_index:
                nonzero_base = base[base != 0]
                if len(nonzero_base) == 0:
                    continue

                counter = Counter(nonzero_base.tolist())
                dominant_value, _ = counter.most_common(1)[0]
                dominant_value = float(dominant_value)

                mse_input = (nonzero_base - dominant_value) ** 2
            else:
                # Pomijamy tylko pozycje, gdzie obie wartości są zerowe
                mask = ~((base == 0) & (compare_tensor == 0))
                if not torch.any(mask):
                    continue

                mse_input = (base[mask] - compare_tensor[mask]) ** 2

            mse = torch.mean(mse_input)
            result += mse / num_channels

        return result/analized_data_lenght

    def _calculate_mark_based_on_similar_repetition_distance_v2(
        self, 
        input_data: torch.Tensor, 
        base_channel_index: int, 
        analized_data_length: int
    ) -> float:
        """
        Calculates a score based on repetition patterns in the input tensor [channels, kernels].
        For the base channel, computes the MSE of non-zero values relative to the mode.
        For other channels, computes MSE between non-zero values and the base channel.

        Args:
            input_data (torch.Tensor): Input tensor of shape [channels, kernels]
            base_channel_index (int): Index of the base channel
            analized_data_length (int): Length of analyzed data

        Returns:
            float: Normalized MSE-based score
        """
        input_data = input_data.float()
        base = input_data[base_channel_index]
        num_channels = input_data.shape[0]
        result = torch.tensor(0.0, device=input_data.device)

        # Pre-compute mask for non-zero base values to avoid redundant computation
        nonzero_mask = base != 0
        
        for channel_index in range(num_channels):
            compare_tensor = input_data[channel_index]
            
            if channel_index == base_channel_index:
                if not nonzero_mask.any():
                    continue
                    
                # Use torch.mode for faster mode calculation
                dominant_value = torch.mode(base[nonzero_mask])[0].float()
                
                mse_input = (base[nonzero_mask] - dominant_value) ** 2
            else:
                # Combined mask for non-zero values in either tensor
                valid_mask = nonzero_mask | (compare_tensor != 0)
                if not valid_mask.any():
                    continue
                    
                mse_input = (base[valid_mask] - compare_tensor[valid_mask]) ** 2
            
            result += mse_input.mean() / num_channels
        
        return result / analized_data_length
    
    def _apply_kerneled_mse_on_one_series(self, series: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
        """
        Applies kernel-based MSE comparison on a single series.

        Args:
            series (torch.Tensor): 1D input series
            kernels (torch.Tensor): Shape [num_kernels, 1, kernel_length]

        Returns:
            torch.Tensor: MSE values [num_kernels, sequence_length]
        """
        kernel_length = kernels.shape[-1]
        padded_series = F.pad(series, (kernel_length // 2, kernel_length // 2), "constant", 0)
        unfolded_series = padded_series.unfold(0, kernel_length, 1).unsqueeze(0)
        mse = (kernels - unfolded_series).abs().mean(dim=2)
        
        if mse.shape[1] < series.shape[0]:
            mse = F.pad(mse, (0, series.shape[0] - mse.shape[1]), value=1)
        return mse[:, :series.shape[0]]
        
    
    
if __name__ == "__main__":
    #region konfiguracja loggera
    logger = logging.getLogger("my_debug_logger")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.propagate = False
    #endregion
    
    comparer = TimeSeriesMovedInTimeComparer(TimeSeriesMovedInTimeComparer.ComparisonMode.DOMINANT, logger= logger)
    
    #region inicjalizacja danych symulacyjnych
    dset = LocalizationDataFormat(r'C:\Users\Marcin\Desktop\Studia\Praca_dyplomowa\Wersja_styczniowa\LeakDetection\data\localization\v2_samples1000_lenght22_typeLocalisation.npz')
    train_loader = DataLoader(dataset=dset, batch_size=2, shuffle=False, pin_memory=True)
    for batch, (real_data, real_leak_label, real_localization) in enumerate(train_loader):
        real_data = real_data.reshape(real_data.size(0), -1, real_data.size(3))
        real_data = real_data.permute(0, 2, 1)
    #endregion 
        start_time = time()
        comparer.compare(real_data)
    
        stop_time = time()
        print(stop_time-start_time)
            