import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datasets import LocalizationDataFormat
from comet_ml import Experiment
from tslearn.metrics import dtw
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import uuid
import random
import os

# Set random seed for reproducibility
torch.manual_seed(42)

print(torch.version.cuda)       # Powinno zwrócić np. '11.8'
print(torch.backends.cudnn.enabled)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Earlystopping
class AE_EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
    
class GAN_EarlyStopping:
    def __init__(self, patience=5, high_loss_threshold=0.8):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            high_loss_threshold (float): Maximum loss value find as bad.
        """
        self.patience = patience
        self.high_loss_threshold = high_loss_threshold
        self.counter = 0
        self.should_stop = False
    
    def step(self, g_loss, d_loss):
        if g_loss > self.high_loss_threshold or d_loss > self.high_loss_threshold:
            self.counter += 1
        else:
            self.counter = 0  # reset if losses drop below threshold

        if self.counter >= self.patience:
            self.should_stop = True
        return self.should_stop


# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim = 100, init_size = 3, noise_strength = 0.05, num_channels = 4):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.init_size = init_size
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
    def __init__(self, num_channels = 4):
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
    def __init__(self, generator, discriminator, experiment, path):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.experiment = experiment
        self.mean_values = torch.tensor([0.0, -0.32172874023188663, 0.9329161398211201, 1.050562329499409]).to(device)
        self.path = path
        self.day_timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
                
    #region GAN            
    def fit_GAN(self, train_dataloader, val_dataloader, num_epochs, learning_rate, patience = 50, high_loss_threshold = 0.8, G_D_ratio = 4):
        adversarial_loss = nn.BCEWithLogitsLoss()
        earlystopper = GAN_EarlyStopping(patience, high_loss_threshold)
        optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate*0.1, betas=(0.5, 0.999))
        
        self.num_epochs = num_epochs
        for epoch in range(num_epochs):
            # train phase
            self._GAN_train_phase(train_dataloader, optimizer_G, optimizer_D, adversarial_loss, epoch, G_D_ratio)
            # validation phase
            val_g_loss, val_d_loss = self._GAN_val_phase(val_dataloader, optimizer_G, optimizer_D, adversarial_loss, epoch)
            # early stopping phase
            if earlystopper.step(val_g_loss, val_d_loss):
                print(f"GAN: Training stopped due to high loss for {patience} consecutive epochs.")
                
                save_dir = os.path.join(self.path, self.day_timestamp, "GAN")
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, "best_generator_GAN_phase.pt")
                torch.save(self.generator.state_dict(), save_path)
                break

        print(f"GAN: pass all planned steps.")
        
        save_dir = os.path.join(self.path, self.day_timestamp, "GAN")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, "last_generator_GAN_phase.pt")
        torch.save(self.generator.state_dict(), save_path)

    def _GAN_train_phase(self, dataloader, optimizer_G, optimizer_D, loss_fn, epoch, G_D_ratio):
        self.generator.train()
        self.discriminator.train()

        total_g_loss = 0.0
        total_d_loss = 0.0

        for i, (data, label, localization) in enumerate(dataloader):
            # Prepare real data
            data = data.reshape(data.size(0), -1, data.size(3))
            data = data.permute(0, 2, 1)
            real_data = data.to(device).float()
            normalized_input_data, normalizing_factor = self.normalize(real_data)
            batch_size_current = normalized_input_data.size(0)
            
            # Labels
            real_label = torch.ones(batch_size_current, 1).to(device)
            fake_label = torch.zeros(batch_size_current, 1).to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            real_output = self.discriminator(normalized_input_data)
            d_loss_real = loss_fn(real_output, real_label)
            
            z = torch.stack([label, localization], dim=1).float()
            z = z.to(device)

            fake_data = self.generator(z)

            fake_output = self.discriminator(fake_data.detach())
            d_loss_fake = loss_fn(fake_output, fake_label)
            
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            total_d_loss += d_loss.item()
            optimizer_D.step()
            
            # Train Generator
            for _ in range(G_D_ratio):
                optimizer_G.zero_grad()
                fake_data = self.generator(z)  # Regenerate fake data for each step if needed
                fake_output = self.discriminator(fake_data)
                g_loss = loss_fn(fake_output, real_label)
                g_loss.backward()
                total_g_loss += g_loss.item()
                optimizer_G.step()
            
            # Log losses
            self.experiment.log_metric("D_loss", d_loss.item(), step=epoch * len(dataloader) + i)
            self.experiment.log_metric("G_loss", g_loss.item(), step=epoch * len(dataloader) + i)
            

            # Log generated samples every 100 iterations
            if i % 100 == 0:
                with torch.no_grad():
                    sample = self.generator(torch.randn(1, 2).to(device))
                    sample = sample[0].detach().cpu().numpy()
                    self.log_data_as_plot(sample, epoch, i, "generated")

        total_g_loss = total_g_loss/(len(dataloader)*G_D_ratio)
        total_d_loss = total_d_loss/(len(dataloader)*G_D_ratio)
        
        print(f"Epoch [{epoch+1}/{self.num_epochs}] D_loss: {total_d_loss:.4f} G_loss: {total_g_loss:.4f}")

    def _GAN_val_phase(self, dataloader, optimizer_G, optimizer_D, loss_fn, epoch):
        self.generator.eval()
        self.discriminator.eval()

        total_g_loss = 0.0
        total_d_loss = 0.0

        for i, (data, label, localization) in enumerate(dataloader):
            # Prepare real data
            data = data.reshape(data.size(0), -1, data.size(3))
            data = data.permute(0, 2, 1)
            real_data = data.to(device).float()
            normalized_input_data, normalizing_factor = self.normalize(real_data)
            batch_size_current = real_data.size(0)
            
            # Labels
            real_label = torch.ones(batch_size_current, 1).to(device)
            fake_label = torch.zeros(batch_size_current, 1).to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            real_output = self.discriminator(normalized_input_data)
            d_loss_real = loss_fn(real_output, real_label)


            z = torch.stack([label, localization], dim=1).float()
            z = z.to(device)

            fake_data = self.generator(z)

            fake_output = self.discriminator(fake_data.detach())
            d_loss_fake = loss_fn(fake_output, fake_label)
            
            d_loss = (d_loss_real + d_loss_fake) / 2
            total_d_loss += d_loss.item()

            
            # Train Generator
            optimizer_G.zero_grad()
            fake_output = self.discriminator(fake_data)
            g_loss = loss_fn(fake_output, real_label)
            total_g_loss += g_loss.item()
            
            # Log losses
            self.experiment.log_metric("D_loss", d_loss.item(), step=epoch * len(dataloader) + i)
            self.experiment.log_metric("G_loss", g_loss.item(), step=epoch * len(dataloader) + i)
            
            # Log generated samples every 100 iterations
            if i % 100 == 0:
                with torch.no_grad():
                    sample = self.generator(torch.randn(1, 2).to(device))
                    sample = sample[0].detach().cpu().numpy()
                    self.log_data_as_plot(sample, epoch, i, "generated")

        total_g_loss = total_g_loss/len(dataloader)
        total_d_loss = total_d_loss/len(dataloader)

        print(f"Epoch [{epoch+1}/{self.num_epochs}] D_loss: {total_d_loss:.4f} G_loss: {total_g_loss:.4f}")
        return total_g_loss, total_d_loss

    #endregion

    #region AE
    def fit_AE(self, train_dataloader, validation_dataloader, num_epochs, learning_rate, patience = 50, min_delta = 0.001):
        reconstruction_loss_fn = nn.MSELoss()
        early_stopping = AE_EarlyStopping(patience=patience, min_delta=min_delta)
        optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs

        for epoch in range(num_epochs):
            # train phase
            self._AE_train_phase(train_dataloader, optimizer, reconstruction_loss_fn, epoch)
            # validation phase
            val_loss = self._AE_validation_phase(validation_dataloader, reconstruction_loss_fn, epoch)
            # early stopping phase
            if early_stopping.step(val_loss):
                print("AE: Early stopping triggered. Training stopped")
                print(f"✅ Model saved with val_loss = {early_stopping.best_loss:.4f}")

                save_dir = os.path.join(self.path, self.day_timestamp, "AE")
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, "best_generator_AE_phase.pt")
                torch.save(self.generator.state_dict(), save_path)
                break
        print("AE: Passed all planned steps.")
        print(f"✅ Model saved with val_loss = {early_stopping.best_loss:.4f}")

        save_dir = os.path.join(self.path, self.day_timestamp, "AE")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, "last_generator_AE_phase.pt")
        torch.save(self.generator.state_dict(), save_path)

            
    def _AE_train_phase(self, dataloader, optimizer, loss_fn, epoch):
        total_loss = 0
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
            loss = self._calculate_AE_loss(reconstructed, input_data, loss_fn)

            total_loss += loss.item()
            
            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Log loss
            self.experiment.log_metric("MSE_AE_train_Loss", loss.item(), step=epoch * len(dataloader) + i)
            # Log reconstructed sample every 100 iterations
            if i % 100 == 0:
                with torch.no_grad():
                    normalized_sample_input = normalized_input_data[0].detach().cpu().numpy()
                    sample_recon = reconstructed[0].detach().cpu().numpy()
                    self.log_data_as_plot(sample_recon, epoch, i, "reconstruction")
                    self.log_data_as_plot(normalized_sample_input, epoch, i, "normalized_sample")
        total_loss = total_loss/len(dataloader)
        print(f"Epoch [{epoch+1}/{self.num_epochs}] Reconstruction_AE_train_Loss: {total_loss:.4f}")
        
    def _AE_validation_phase(self, dataloader, loss_fn, epoch):
        total_loss = 0
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
            loss = self._calculate_AE_loss(reconstructed, input_data, loss_fn)

            total_loss += loss.item()

            # Log loss
            self.experiment.log_metric("Total_AE_val_Loss", loss.item(), step=epoch * len(dataloader) + i)
            self.experiment.log_metric("MSE_AE_val_Loss", loss.item(), step=epoch * len(dataloader) + i)
            self.experiment.log_metric("DTW_AE_val_Loss", loss.item(), step=epoch * len(dataloader) + i)
            # Log reconstructed sample every 100 iterations
            if i % 100 == 0:
                with torch.no_grad():
                    normalized_sample_input = normalized_input_data[0].detach().cpu().numpy()
                    sample_recon = reconstructed[0].detach().cpu().numpy()
                    self.log_data_as_plot(sample_recon, epoch, i, "reconstruction_val")
                    self.log_data_as_plot(normalized_sample_input, epoch, i, "normalized_sample_val")

        total_loss = total_loss/len(dataloader)
        print(f"Epoch [{epoch+1}/{self.num_epochs}] Reconstruction_AE_val_Loss: {total_loss:.4f}")
        return total_loss

    def _calculate_AE_loss(self, output, input, loss_fn):
        loss = loss_fn(output.float(), input)
        return loss
        

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
            batch_data = tensor[b].detach().cpu().numpy()
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
    dataset = LocalizationDataFormat(root_dir=r"data\localization\v2_samples126_lenght22_typeLocalisation.npz")
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    # Initialize CometML
    experiment = Experiment(
        api_key="V6xW30HU42MtnnpSl6bsGODZ1",
        project_name="time-series-gan",
    )
    experiment.set_name("Multi-Channel Time Series GAN")

    # Training loop
    full_model = GAN(generator, discriminator, experiment, "models\GAN\AE_MSE-GAN")
    print("AE")
    full_model.fit_AE(train_dataloader, val_dataloader, 1000, 3e-5)
    print("GAN")
    full_model.fit_GAN(train_dataloader, val_dataloader, 1000, 3e-3, patience= 100)
    
    experiment.end()
#endregion