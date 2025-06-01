from torch.utils.data import TensorDataset, DataLoader, random_split
from datasets import LocalizationDataFormat
import torch
import data_preprocessing
import matplotlib.pyplot as plt

def plot_vectors(original: torch.Tensor, restored: torch.Tensor):
    """Plots all four original and restored vectors."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    for i in range(original.shape[0]):
        plt.plot(original[i].numpy(), label=f'Original {i}')
    plt.title("Original Vectors")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for i in range(restored.shape[0]):
        plt.plot(restored[i].numpy(), label=f'Restored {i}')
    plt.title("Restored Vectors")
    plt.legend()
    
    plt.tight_layout()
    plt.show()


dset = LocalizationDataFormat(r'C:\Users\Marcin\Desktop\Studia\Praca_dyplomowa\Wersja_styczniowa\LeakDetection\data\localization\v2_samples1000_lenght22_typeLocalisation.npz')
train_size = int(0.8 * len(dset))
validation_size = 2*(len(dset) - train_size)//3
test_size = len(dset) - validation_size - train_size
train_dataset_1, validation_dataset_1, test_dataset_1 = random_split(dataset=dset, lengths=[train_size, validation_size, test_size], generator=torch.Generator().manual_seed(42))  # fix the generator for reproducible results

train_loader = DataLoader(dataset=train_dataset_1, batch_size=1, shuffle=True, pin_memory=True)

for batch, (real_data, real_leak_label, real_localization) in enumerate(train_loader):
    real_data = real_data.reshape(real_data.size(0), -1, real_data.size(3))
    real_data = real_data.permute(0, 2, 1)
    fourier_data = data_preprocessing.vector_to_fourier(real_data.squeeze())
    restored_data = data_preprocessing.fourier_to_vector(fourier_data)
    print(torch.allclose(real_data.squeeze(), restored_data, atol=1e-6))  # Should return True
    print(real_data.squeeze().shape)
    print(fourier_data.shape)
    print(restored_data.shape)
    plot_vectors(real_data.squeeze(), fourier_data)
    a = input("dupa")