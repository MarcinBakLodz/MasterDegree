import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import LocalizationDataFormat
from comet_ml import Experiment
import numpy as np
import uuid

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

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.init_size = 3
        self.fc = nn.Linear(latent_dim, 512 * self.init_size)
        
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
    
    def forward(self, z):
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

if __name__ == "__main__":
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Loss and optimizers
    adversarial_loss = nn.BCEWithLogitsLoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Data loading
    dataset = LocalizationDataFormat(root_dir="/home/mbak/LeakDetection/data/localization/v2_samples126_lenght22_typeLocalisation.npz")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize CometML
    experiment = Experiment(
        api_key="V6xW30HU42MtnnpSl6bsGODZ1",
        project_name="time-series-gan",
    )
    experiment.set_name("Multi-Channel Time Series GAN")

    # Training loop
    for epoch in range(num_epochs):
        for i, (data, _, _) in enumerate(dataloader):
            # Prepare real data
            real_data = data.to(device).float()
            batch_size_current = real_data.size(0)
            
            # Labels
            real_label = torch.ones(batch_size_current, 1).to(device)
            fake_label = torch.zeros(batch_size_current, 1).to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            real_output = discriminator(real_data)
            d_loss_real = adversarial_loss(real_output, real_label)
            
            z = torch.randn(batch_size_current, latent_dim).to(device)
            fake_data = generator(z)
            fake_output = discriminator(fake_data.detach())
            d_loss_fake = adversarial_loss(fake_output, fake_label)
            
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            fake_output = discriminator(fake_data)
            g_loss = adversarial_loss(fake_output, real_label)
            g_loss.backward()
            optimizer_G.step()
            
            # Log losses
            experiment.log_metric("D_loss", d_loss.item(), step=epoch * len(dataloader) + i)
            experiment.log_metric("G_loss", g_loss.item(), step=epoch * len(dataloader) + i)
            
            # Log generated samples every 100 iterations
            if i % 100 == 0:
                with torch.no_grad():
                    sample = generator(torch.randn(1, latent_dim).to(device)).cpu().numpy()
                    experiment.log_image(sample[0, 0, :], name=f"epoch_{epoch}_iter_{i}_channel_0.png")
        
        print(f"Epoch [{epoch+1}/{num_epochs}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

    # End experiment
    experiment.end()