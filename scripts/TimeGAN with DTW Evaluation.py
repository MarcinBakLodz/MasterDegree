import numpy as np
import pandas as pd
from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
from ydata_synthetic.preprocessing.timeseries import processed_stock
from dtw import dtw
from sklearn.preprocessing import MinMaxScaler

# Define model parameters
gan_args = ModelParameters(batch_size=128, lr=5e-4, noise_dim=32, layers_dim=128, latent_dim=24, gamma=1)
train_args = TrainParameters(epochs=5000, sequence_length=24, number_sequences=6)

# Load and preprocess your multichannel time series data
# Example: Replace with your own dataset
data = pd.read_csv("your_data.csv")  # Shape: (n_samples, n_channels)
cols = list(data.columns)

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Augment data with time shifts
shifted_data = []
for shift in [-10, -5, 0, 5, 10]:  # Example shifts
    shifted = np.roll(data_scaled, shift, axis=0)
    shifted_data.append(shifted)
augmented_data = np.concatenate(shifted_data, axis=0)

# Convert to DataFrame
augmented_df = pd.DataFrame(augmented_data, columns=cols)

# Train TimeGAN
synth = TimeSeriesSynthesizer(modelname='timegan', model_parameters=gan_args)
synth.fit(augmented_df, train_args, num_cols=cols)
synth.save('synthesizer.pkl')

# Generate synthetic data
synth_data = synth.sample(n_samples=len(data))

# Evaluate using DTW
def compute_dtw(real, synth):
    distances = []
    for r, s in zip(real, synth):
        dist, _, _ = dtw(r, s, dist=lambda x, y: np.linalg.norm(x - y))
        distances.append(dist)
    return np.mean(distances)

dtw_score = compute_dtw(data_scaled, synth_data)
print(f"Average DTW Distance: {dtw_score}")