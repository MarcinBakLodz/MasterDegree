import torch
import numpy as np
from tslearn.metrics import dtw

def channel_dtw_similarity(tensor):
    """
    Compute average DTW similarity between channels in a tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape [batch, channels, time]
    
    Returns:
        torch.Tensor: Tensor of shape [batch] with average DTW distances
    """
    batch_size, num_channels, time_length = tensor.shape
    result = torch.zeros(batch_size)
    
    for b in range(batch_size):
        batch_data = tensor[b].cpu().numpy()
        dtw_sum = 0.0
        pair_count = 0
        
        for i in range(num_channels):
            for j in range(i + 1, num_channels):
                dtw_dist = dtw(batch_data[i], batch_data[j])
                dtw_sum += dtw_dist
                pair_count += 1
        
        result[b] = dtw_sum / pair_count if pair_count > 0 else 0.0
    
    return result

# Funkcje testujące
def test_channel_dtw_similarity():
    # Przygotowanie danych testowych
    def test_case_1():
        """Test 1: Standardowy przypadek z 2 partiami, 3 kanałami i sekwencją czasową."""
        tensor = torch.tensor([
            [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [2.0, 4.0, 6.0]],  # Batch 1
            [[0.0, 1.0, 0.0], [0.1, 1.1, 0.1], [1.0, 2.0, 1.0]],   # Batch 2
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]   # Batch 3
        ])
        result = channel_dtw_similarity(tensor)
        print("Test 1 - Standardowy przypadek:", result)
        assert result.shape == (3,), "Nieprawidłowy kształt wyniku"
        assert all(result >= 0), "DTW nie powinno zwracać wartości ujemnych"

    def test_case_2():
        """Test 2: Pojedynczy kanał w partii (powinien zwrócić 0.0)."""
        tensor = torch.randn(2, 1, 50)  # 2 partie, 1 kanał, 50 próbek czasowych
        result = channel_dtw_similarity(tensor)
        print("Test 2 - Pojedynczy kanał:", result)
        assert result.shape == (2,), "Nieprawidłowy kształt wyniku"
        assert torch.all(result == 0.0), "Wynik powinien być 0 dla pojedynczego kanału"

    def test_case_3():
        """Test 3: Identyczne kanały (DTW powinno zwrócić ~0 dla identycznych sygnałów)."""
        tensor = torch.ones(1, 2, 10)  # 1 partia, 2 identyczne kanały
        result = channel_dtw_similarity(tensor)
        print("Test 3 - Identyczne kanały:", result)
        assert result.shape == (1,), "Nieprawidłowy kształt wyniku"
        assert abs(result[0]) < 1e-5, "DTW dla identycznych kanałów powinno być bliskie 0"

    def test_case_4():
        """Test 4: Losowe dane i weryfikacja spójności."""
        tensor = torch.randn(3, 4, 100)  # 3 partie, 4 kanały, 100 próbek
        result = channel_dtw_similarity(tensor)
        print("Test 4 - Losowe dane:", result)
        assert result.shape == (3,), "Nieprawidłowy kształt wyniku"
        assert all(result >= 0), "DTW nie powinno zwracać wartości ujemnych"

    def test_case_5():
        """Test 5: Pusty tensor (powinien zgłosić błąd)."""
        tensor = torch.randn(0, 2, 10)  # Pusty batch
        try:
            result = channel_dtw_similarity(tensor)
            assert False, "Funkcja powinna zgłosić błąd dla pustego tensora"
        except Exception as e:
            print("Test 5 - Pusty tensor: Poprawnie zgłoszono błąd", e)

    # Uruchomienie testów
    print("Uruchamianie testów...")
    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4()
    test_case_4()
    test_case_5()
    print("Wszystkie testy zakończone pomyślnie!")

# Uruchom testy
if __name__ == "__main__":
    test_channel_dtw_similarity()