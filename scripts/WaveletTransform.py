""" Celem tego pliku jest opracowanie metody porównywania przebiegów z rurociągów.
    Plan jak ma to działać:
        1) Wyrównanie przebiegów z różnych manometrów do jednej wysokości o stałą wartość.
        1) Podział przebiegu na kernele - done
        2) Konwolucja kernelami - done
        3) 
"""

import logging
from datasets import LocalizationDataFormat
import torch.utils.data as torch_utils_data
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import torch.nn.functional as F
import time
from statistics import mean
from collections import Counter
from enum import Enum


# Konfiguracja loggera
logger = logging.getLogger("my_debug_logger")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.propagate = False

class ComparisonMode(Enum):
    MEAN = 0,
    DOMINANT = 1

def change_one_series_to_kernels(series: torch.Tensor, size: int, bypass: tuple = (0, 0)) -> torch.Tensor:
    """
    Dzieli jednowymiarowy tensor na mniejsze fragmenty o zadanym rozmiarze (kernelach).

    Args:
        series (torch.Tensor): Jednowymiarowy tensor danych wejściowych.
        size (int): Rozmiar każdego kernela.
        bypass (tuple, optional): Liczba elementów do pominięcia na początku i końcu serii. Domyślnie (0, 0).

    Returns:
        torch.Tensor: Tensor o kształcie (N, 1, size), gdzie N to liczba wyodrębnionych kernelów.
    """
    logger.debug(f"\t\t\tchange_one_series_to_kernels - wejście: size={size}, bypass={bypass}")
    analized_series = series[bypass[0]:-bypass[1]] if sum(bypass) > 0 else series
    number_of_kernels = analized_series.shape[-1] // size

    result = []
    for i in range(number_of_kernels):
        kernel = analized_series[i * size: i * size + size]
        result.append(kernel)

    result = torch.stack(result).unsqueeze(1)
    logger.debug(f"\t\t\tchange_one_series_to_kernels - liczba kernelów: {number_of_kernels}, kształt wyniku: {result.shape}")
    return result


def apply_kerneled_mse_on_one_series(series: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    """
    Oblicza średni błąd bezwzględny (MSE) pomiędzy fragmentami serii a zdefiniowanymi kernelami.

    Args:
        series (torch.Tensor): Jednowymiarowy tensor danych wejściowych.
        kernels (torch.Tensor): Tensor z kernelami (kształt: N, 1, size).

    Returns:
        torch.Tensor: Tensor podobieństw (shape: N, L), gdzie L to długość serii z uwzględnieniem paddingu.
    """
    
    logger.debug(f"\t\t\tapply_kerneled_mse_on_one_series - wejście: series.shape={series.shape}, kernels.shape={kernels.shape}")
    kernel_lenght = kernels.shape[-1]

    # Padding umożliwia przesuwanie kernela po serii
    analized_series = F.pad(series, (kernel_lenght // 2, kernel_lenght // 2), "constant", 0)

    result = []
    for i in range(analized_series.shape[-1] - kernel_lenght):
        part_of_analized_series = analized_series[i:i + kernel_lenght]

        # Obliczenie błędu bezwzględnego między fragmentem a wszystkimi kernelami
        analized_sum = abs(kernels - part_of_analized_series)
        analized_sum = analized_sum.sum(dim=2) / kernel_lenght  # Średnia błędu
        analized_sum = analized_sum.squeeze(1)

        result.append(analized_sum)

    result = torch.stack(result, dim=1)

    # Uzupełnienie wyników zerami, jeśli jest krótszy niż oryginalna seria
    if result.shape[-1] < series.shape[-1]:
        pad_size = series.shape[-1] - result.shape[-1]
        result = F.pad(result, (0, pad_size), value=1)
        logger.debug(f"\t\t\tapply_kerneled_mse_on_one_series - padding applied: pad_size={pad_size}")

    logger.debug(f"\t\t\tapply_kerneled_mse_on_one_series - wynikowy kształt: {result.shape}")
    return result

def find_similar_regions(mse_result: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Zwraca tensor binarny z informacjami które obszary zostały zakwalifikowane jako podobne

    Args:
        mse_result (torch.Tensor): Tensor o kształcie (N, L), wynik działania apply_kerneled_mse_on_one_series.
                                   N to liczba kernelów, L to długość serii.
        threshold (float): Wartość progowa, poniżej której traktujemy fragmenty jako podobne.

    Returns:
        torch.Tensor: Tensor typu bool o tym samym kształcie (N, L), gdzie True oznacza spełnienie warunku.
    """
    return mse_result < threshold

def get_true_indices_per_row(bool_tensor: torch.Tensor) -> list[list[int]]:
    """
    Zwraca listę N list, gdzie każda zawiera indeksy elementów równych True w danym wierszu.

    Args:
        bool_tensor (torch.Tensor): Tensor typu bool o kształcie (N, L)

    Returns:
        list[list[int]]: Lista N list z indeksami True.
    """
    if bool_tensor.dtype != torch.bool or bool_tensor.ndim != 2:
        raise ValueError("Input tensor must be 2D and of dtype bool")

    return [row.nonzero(as_tuple=True)[0].tolist() for row in bool_tensor]

def get_distances_beetween_similar_points(list_of_similarity_lists: list[list[int]], mode:ComparisonMode) -> list[list[int]]:
    result = []
    result_means = []
    result_dominant = []
    for kernel in list_of_similarity_lists:
        result_kernel = []
        result_kernel_mean = 0
        result_kernel_dominant = 0
        for i in range(len(kernel) - 1):
            result_kernel.append(kernel[i+1] - kernel[i])
        
        if result_kernel:
            result_kernel_mean = mean(result_kernel)
            result_kernel_dominant = Counter(result_kernel).most_common(1)[0][0]
        
        result_means.append(result_kernel_mean)
        result_dominant.append(result_kernel_dominant)
    
    if mode == ComparisonMode.MEAN:
        return result_means
    if mode == ComparisonMode.DOMINANT:
        return result_dominant
    
        
        
    
                                    

if __name__ == "__main__":
    #variables
    analized_point = 30
    kernel_size = 30
    similarity_threshold = 0.0125
    

    # Średnie wartości używane do wyrównania danych
    mean_values = [0.0, -0.32172874023188663, 0.9329161398211201, 1.050562329499409]

    # Wczytanie zbioru danych
    logger.info("Ładowanie zbioru danych...")
    dset = LocalizationDataFormat(
        'C:\\Users\\Marcin\\Desktop\\Studia\\Praca_dyplomowa\\Wersja_styczniowa\\LeakDetection\\data\\localization\\v2_samples1000_lenght22_typeLocalisation.npz'
    )
    data_loader: torch_utils_data.dataloader.DataLoader = torch_utils_data.DataLoader(dataset=dset, batch_size=32, shuffle=False, pin_memory=True)
    logger.info("Dane zostały załadowane.")


    for real_data, real_leak_label, real_localization in data_loader:
        logger.info("Przetwarzanie batcha danych...")
        start_time = time.time()
        
        
        # Przekształcenie danych do kształtu: (batch_size, channels, length)
        real_data = real_data.reshape(real_data.size(0), -1, real_data.size(3))
        real_data = real_data.permute(0, 2, 1)
        logger.debug(f"\tbatch: {real_data.shape}")

        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        axs[0].set_ylim(-5, 5)
        axs[0].set_title("Sygnał wejściowy po wyrównaniu")
        axs[1].set_title("Podobieństwo (średni błąd)")

        axs[1].set_xlabel("Pozycja w czasie")
        axs[0].set_ylabel("Amplituda")
        axs[1].set_ylabel("Błąd")
        axs[0].axvspan(analized_point*kernel_size-kernel_size//2, analized_point*kernel_size+kernel_size//2, color='yellow', alpha=0.3)
        axs[1].axvspan(analized_point*kernel_size-kernel_size//2, analized_point*kernel_size+kernel_size//2, color='yellow', alpha=0.3)
        axs[2].axvspan(analized_point*1, analized_point+1, color='yellow', alpha=0.3)
        
        
        constant_shifting_y = real_data[0, 0] - mean_values[0] - torch.mean(real_data[0, 0])
        custom_kernel = change_one_series_to_kernels(constant_shifting_y, kernel_size, (10, 10))
        
        for channel in range(4):
            similarity_indexes_per_channel = []
            logger.info(f"\tchannel {channel}:")

            # Usunięcie średniej i przesunięcie kanału
            constant_shifting_y = real_data[0, channel] - mean_values[channel] - torch.mean(real_data[0, 0])
            # Tworzenie kernelów z kanału czasowego
            logger.debug(f"\t\tkernels: {custom_kernel.shape}")

            # Obliczanie podobieństwa przez kernele
            similarity_value = apply_kerneled_mse_on_one_series(constant_shifting_y, custom_kernel)
            logger.debug(f"\t\tpodobieństwo serii danych: {similarity_value.shape}")
            similarity_points = find_similar_regions(similarity_value, similarity_threshold)
            logger.debug(f"\t\tdostatecznie podobne serie danych: {similarity_points.shape}")
            indexes_of_similar_points = get_true_indices_per_row(similarity_points)
            similarity_indexes_per_channel.append(indexes_of_similar_points)
            distance_beetween_similar_points = get_distances_beetween_similar_points(indexes_of_similar_points, ComparisonMode.MEAN)
            logger.debug(f"distance_beetween_similar_points: {distance_beetween_similar_points}")
            
            # Wykresy wyników
            axs[0].plot(constant_shifting_y, label=f'Kanał {channel}', linewidth=1.5)
            axs[1].plot(similarity_value[analized_point], label=f'Kanał {channel}', linewidth=1.5)
            axs[2].plot(similarity_points[analized_point], label=f'Kanał {channel}', linewidth=1.5)
        

        end_time = time.time()


        for ax in axs:
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.suptitle("Analiza sygnału – konwolucja i podobieństwo", fontsize=16, y=1.02)
        plt.show()
    
