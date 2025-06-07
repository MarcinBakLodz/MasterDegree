import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datasets import LocalizationDataFormat
from comet_ml import Experiment
from tslearn.metrics import dtw
import matplotlib.pyplot as plt
import numpy as np
import uuid
import random

# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 100
num_epochs = 100
batch_size = 32
sequence_length = 2220
num_channels = 4
learning_rate = 0.0002
noise_strength = 0.05


# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.init_size = 3
        self.embeder = nn.Linear(2, latent_dim)
        self.fc = nn.Linear(latent_dim, 512 * self.init_size)
        self.noise_strength = noise_strength
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=2, padding=0),  # 3 -> 7
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.ConvTranspose1d(512, 512, kernel_size=4, stride=2, padding=0),  # 7 -> 16
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=2, padding=0),  # 16 -> 33
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.ConvTranspose1d(512, 512, kernel_size=4, stride=2, padding=0),  # 33 -> 68
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=2, padding=0),  # 68 -> 137
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.ConvTranspose1d(512, 512, kernel_size=4, stride=2, padding=0),  # 137 -> 276
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.ConvTranspose1d(512, 512, kernel_size=4, stride=2, padding=0),  # 276 -> 554
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=2, padding=0),  # 554 -> 1109
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.ConvTranspose1d(512, num_channels, kernel_size=4, stride=2, padding=0),  # 1109 -> 2220
            nn.Sigmoid()  # Output normalized to [0,1]
        )
    
    def forward(self, x):
        x = self.embeder(x)
        noise = torch.randn_like(x) * self.noise_strength
        z = x + noise
        out = self.fc(z)
        out = out.view(out.size(0), 512, self.init_size)
        out = self.conv_blocks(out)
        return out

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Linear(512, 1)
    
    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class GAN(nn.Module):
    def __init__(self, generator, discriminator, experiment):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.experiment = experiment
        self.mean_values = torch.tensor([0.0, -0.32172874023188663, 0.9329161398211201, 1.050562329499409])
        
                
    #region GAN            
    def fit_GAN(self, dataloader, num_epochs):
        adversarial_loss = nn.BCEWithLogitsLoss()
        optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        
        for epoch in range(num_epochs):
            # train phase
            for i, (data, label, localization) in enumerate(dataloader):
                # Prepare real data
                data = data.reshape(data.size(0), -1, data.size(3))
                data = data.permute(0, 2, 1)
                real_data = data.to(device).float()
                batch_size_current = real_data.size(0)
                
                # Labels
                real_label = torch.ones(batch_size_current, 1).to(device)
                fake_label = torch.zeros(batch_size_current, 1).to(device)
                
                # Train Discriminator
                optimizer_D.zero_grad()
                real_output = self.discriminator(real_data)
                d_loss_real = adversarial_loss(real_output, real_label)
                
                z = torch.stack([label, localization], dim=1).float()
                z = z.to(device)

                fake_data = self.generator(z)

                fake_output = self.discriminator(fake_data.detach())
                d_loss_fake = adversarial_loss(fake_output, fake_label)
                
                d_loss = (d_loss_real + d_loss_fake) / 2
                d_loss.backward()
                optimizer_D.step()
                
                # Train Generator
                optimizer_G.zero_grad()
                fake_output = self.discriminator(fake_data)
                g_loss = adversarial_loss(fake_output, real_label)
                g_loss.backward()
                optimizer_G.step()
                
                # Log losses
                self.experiment.log_metric("D_loss", d_loss.item(), step=epoch * len(dataloader) + i)
                self.experiment.log_metric("G_loss", g_loss.item(), step=epoch * len(dataloader) + i)
                
                # Log generated samples every 100 iterations
                if i % 100 == 0:
                    with torch.no_grad():
                        sample = self.generator(torch.randn(1, 2).to(device)).cpu().numpy()
                        sample = sample[0].detach().cpu().numpy()
                        self.log_data_as_plot(sample, epoch, i, "generated")

            
            print(f"Epoch [{epoch+1}/{num_epochs}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
            # validation phase
            
            # early stopping phase
    #endregion

    #region AE
    def fit_AE(self, train_dataloader, num_epochs):
        reconstruction_loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            # train phase
            self._AE_train_phase(train_dataloader, optimizer, reconstruction_loss_fn, epoch)
            # validation phase
            # early stopping phase
            
    def _AE_train_phase(self, dataloader, optimizer, loss_fn, epoch):
        self.generator.train()
        for i, (data, label, localization) in enumerate(dataloader):
            # Prepare real data
            data = data.reshape(data.size(0), -1, data.size(3))
            data = data.permute(0, 2, 1)
            input_data = data.to(device).float()
            normalized_input_data, normalizing_factor = self.normalize(input_data)
            
            # Forward pass
            optimizer.zero_grad()
            z = torch.stack([label, localization], dim=1).float()
            z = z.to(device)
            reconstructed = self.generator(z)
            # Truncate reconstructed output to match input_data's sequence length
            reconstructed = reconstructed[:, :, :2200]
            mse_loss = loss_fn(reconstructed.float(), input_data)
            dtw_grade = self.channel_dtw_similarity(normalized_input_data.float())
            dtw_loss = loss_fn(dtw_grade, torch.zeros_like(dtw_grade))
            
            total_loss = mse_loss + dtw_loss
            
            # Backward and optimize
            total_loss.backward()
            optimizer.step()

            # Log loss
            self.experiment.log_metric("Total_AE_train_Loss", total_loss.item(), step=epoch * len(dataloader) + i)
            self.experiment.log_metric("MSE_AE_train_Loss", total_loss.item(), step=epoch * len(dataloader) + i)
            self.experiment.log_metric("DTW_AE_train_Loss", total_loss.item(), step=epoch * len(dataloader) + i)
            # Log reconstructed sample every 100 iterations
            if i % 100 == 0:
                with torch.no_grad():
                    print("wszedlem")
                    normalized_sample_input = normalized_input_data[0].detach().cpu().numpy()
                    sample_recon = reconstructed[0].detach().cpu().numpy()
                    self.log_data_as_plot(sample_recon, epoch, i, "reconstruction")
                    self.log_data_as_plot(normalized_sample_input, epoch, i, "normalized_sample")

        print(f"Epoch [{epoch+1}/{num_epochs}] Reconstruction_AE_train_Loss: {total_loss.item():.4f}")
        
    def _AE_validation_phase(self, dataloader, loss_fn, epoch):
        for i, (data, label, localization) in enumerate(dataloader):
            # Prepare real data
            data = data.reshape(data.size(0), -1, data.size(3))
            data = data.permute(0, 2, 1)
            input_data = data.to(device).float()
            normalized_input_data, normalizing_factor = self.normalize(input_data)
            
            # Forward pass
            z = torch.stack([label, localization], dim=1).float()
            z = z.to(device)
            reconstructed = self.generator(z)
            # Truncate reconstructed output to match input_data's sequence length
            reconstructed = reconstructed[:, :, :2200]
            mse_loss = loss_fn(reconstructed.float(), input_data)
            dtw_grade = self.channel_dtw_similarity(normalized_input_data.float())
            dtw_loss = loss_fn.mse_loss(dtw_grade, torch.zeros_like(dtw_grade))
            
            total_loss = mse_loss + dtw_loss
            

            # Log loss
            self.experiment.log_metric("Total_AE_val_Loss", total_loss.item(), step=epoch * len(dataloader) + i)
            self.experiment.log_metric("MSE_AE_val_Loss", total_loss.item(), step=epoch * len(dataloader) + i)
            self.experiment.log_metric("DTW_AE_val_Loss", total_loss.item(), step=epoch * len(dataloader) + i)
            # Log reconstructed sample every 100 iterations
            if i % 100 == 0:
                with torch.no_grad():
                    print("wszedlem")
                    normalized_sample_input = normalized_input_data[0].detach().cpu().numpy()
                    sample_recon = reconstructed[0].detach().cpu().numpy()
                    self.log_data_as_plot(sample_recon, epoch, i, "reconstruction_val")
                    self.log_data_as_plot(normalized_sample_input, epoch, i, "normalized_sample_val")

        print(f"Epoch [{epoch+1}/{num_epochs}] Reconstruction_AE_val_Loss: {total_loss.item():.4f}")
    #endregion  
    
    #region utils  
    def log_data_as_plot(self, data: torch.Tensor, epoch:int, batch:int, name:str = "data")->None:
        name = f"{epoch}/{batch}_{name}"
        plt.figure(figsize=(80, 8))
        for i in range(4):
            plt.plot(data[i], label=f'manometr{i+1}', linestyle='-')
        plt.ylim(-1, 1) 
        self.experiment.log_figure(figure_name= name, figure= plt)
        plt.close()
    
    def channel_dtw_similarity(self, tensor):
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
            batch_data = tensor[b].detach().numpy()
            dtw_sum = 0.0
            pair_count = 0
            
            for i in range(num_channels):
                for j in range(i + 1, num_channels):
                    dtw_dist = dtw(batch_data[i], batch_data[j])
                    dtw_sum += dtw_dist
                    pair_count += 1
            
            result[b] = dtw_sum / pair_count if pair_count > 0 else 0.0
        
        return result

    def normalize(self, x):
        x = x - self.mean_values.view(1, 4, 1)
        max_per_sample = x.amax(dim=(1, 2)) 
        normalization_factor = 1.0 / max_per_sample
        normalized = x * normalization_factor.view(-1, 1, 1)
        return normalized, normalization_factor
    
    def denormalize(self, x, normalization_factor):
        x = x / normalization_factor.view(-1, 1, 1)
        
        x = x + self.mean_values.view(1, 4, 1)
        
        return x
    #endregion

#region main
if __name__ == "__main__":
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)



    # Data loading
    dataset = LocalizationDataFormat(root_dir=r"C:\Users\Marcin\Desktop\Studia\Praca_dyplomowa\Wersja_czerwcowa\MasterDegree\data\localization\v2_samples126_lenght22_typeLocalisation.npz")
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Initialize CometML
    experiment = Experiment(
        api_key="V6xW30HU42MtnnpSl6bsGODZ1",
        project_name="time-series-gan",
    )
    experiment.set_name("Multi-Channel Time Series GAN")

    # Training loop
    full_model = GAN(generator, discriminator, experiment)
    full_model.fit_AE(dataloader, 1000)
    full_model.fit_GAN(dataloader, 1000)
    
    experiment.end()
#endregion