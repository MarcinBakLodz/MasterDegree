import torch
from collections import Counter

def calculate_mark_based_on_similar_repetition_counter(input_data: torch.Tensor, base_channel_index: int) -> float:
    """
    Oblicza wartość na podstawie podobieństwa powtórzeń cech, z pominięciem wartości zerowych.
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
                print(f"Pomijam channel: {channel_index}")
                continue  # brak danych niezerowych
            print(f"Dla Channela {channel_index}, będą analizowane element {nonzero_base}")

            counter = Counter(nonzero_base.tolist())
            dominant_value, _ = counter.most_common(1)[0]
            dominant_value = float(dominant_value)

            mse_input = (nonzero_base - dominant_value) ** 2
        else:
            # Pomijamy tylko pozycje, gdzie obie wartości są zerowe
            mask = ~((base == 0) & (compare_tensor == 0))
            if not torch.any(mask):
                print(f"Pomijam channel: {channel_index}")
                continue  # brak wspólnych niezerowych danych

            print(f"Dla Channela {channel_index}, będą analizowane element {mask}")
            print(f"\tbase: {base[mask]}")
            print(f"\tcompare: {compare_tensor[mask]}")
            mse_input = (base[mask] - compare_tensor[mask]) ** 2

        mse = torch.mean(mse_input)
        result += mse / num_channels

    return result
    
    # input_data = input_data.float()
    # base = input_data[base_channel_index]
    # result = 0.0
    # num_channels = input_data.shape[0]

    # for channel_index in range(num_channels):
    #     compare_tensor = input_data[channel_index]

    #     if channel_index == base_channel_index:
    #         nonzero_base = base[base > 1]
    #         if len(nonzero_base) == 0:
    #             print(f"Pomijam channel: {channel_index}")
    #             continue  # brak danych niezerowych
    #         print(f"Dla Channela {channel_index}, będą analizowane element {nonzero_base}")

    #         counter = Counter(nonzero_base.tolist())
    #         dominant_value, _ = counter.most_common(1)[0]
    #         dominant_value = float(dominant_value)

    #         mse_input = (nonzero_base - dominant_value) ** 2
    #     else:
    #         # Pomijamy tylko te pozycje, gdzie obie są zerami
    #         mask = ~((base <= 1) & (compare_tensor <= 1))
    #         if not torch.any(mask):
    #             print(f"Pomijam channel: {channel_index}")
    #             continue  # brak wspólnych niezerowych danych

    #         print(f"Dla Channela {channel_index}, będą analizowane element {mask}")
    #         print(f"\tbase: {base[mask]}")
    #         print(f"\tcompare: {compare_tensor[mask]}")
    #         mse_input = (base[mask] - compare_tensor[mask]) ** 2

    #     mse = torch.mean(mse_input)
    #     result += mse / num_channels

    # return result    

# === PRZYKŁADOWE DANE TESTOWE ===
input_data = torch.tensor([
    [3, 0, 1, 1, 3],  # channel 0 (base)
    [3, 1, 3, 3, 3],  # channel 1
    [0, 0, 0, 1, 0],  # channel 2 (same length, all zero)
    [3, 0, 4, 3, 0],  # channel 3 (some overlap)
])

# Wywołanie funkcji
base_channel_index = 3
result = calculate_mark_based_on_similar_repetition_counter(input_data, base_channel_index)

print(f"Wynik (MSE z pominięciem zer): {result}")
