from comet_ml import Experiment
from layers import Discriminator, Generator
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datasets import LocalizationDataFormat
import random
from datetime import datetime
import os

class SynteticDataGenerator(nn.Module):
    def __init__(self, generator, discriminator, experiment:Experiment, batch_size:int = 8, learning_rate:float = 3e-3, ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.learning_rate:float = learning_rate
        self.patience:int = 0
        self.best_accuracy = None
        self.epochs_no_improve:int = 0
        self.number_of_epochs:int = 0
        self.log_validation_phase_loss:bool = True
        self.main_dir:str = "/home/mbak/LeakDetectionwithGit/LeakDetection/models/GAN"
        self.is_test_saving:bool = True
        
        #hyperparameters
        self.batch_size = batch_size
        self.experiment = experiment
        self.min_delta:float = 0.02
        self.patience:int = 5
        self.are_smoothed_labels:bool = True
        
        self.criterion = nn.BCELoss()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.optimizer_D = optim.SGD(self.discriminator.parameters(), lr=self.learning_rate)
        
    def get_dset(self, root_dir, train_share=0.8):
        dset = LocalizationDataFormat(root_dir)
        train_size = int(train_share * len(dset))
        validation_size = 2*(len(dset) - train_size)//3
        test_size = len(dset) - validation_size - train_size
        return random_split(dataset=dset, lengths=[train_size, validation_size, test_size], generator=torch.Generator().manual_seed(42))  # fix the generator for reproducible results
    
    def train(self, train_dataloader:DataLoader, validation_dataloader:DataLoader,  num_of_epochs:int =10):
        
        if self.is_test_saving: self.test_saving()
        
        self.number_of_epochs = num_of_epochs
        self.number_of_batches = len(train_dataloader)
        self.validation_dataloader = validation_dataloader
        if self.experiment: self.log_hyperparameters()
        for epoch in range(self.number_of_epochs):
            for batch, (real_data, real_leak_label, real_localization) in enumerate(train_dataloader):
                real_data = real_data.reshape(real_data.size(0), -1, real_data.size(3))
                real_data = real_data.permute(0, 2, 1)
                real_d_loss = self.real_phase_train(real_data)
                generated_data, fake_d_loss, g_loss = self.fake_phase_train(real_data.size(0))
                
                self.log_losses(epoch, batch, real_d_loss, fake_d_loss, g_loss)     
                self.log_data_as_plot(real_data, epoch, batch, "real_data")        
                self.log_data_as_plot(generated_data, epoch, batch, "generated_data") 
            
            validation_loss =self.phase_validation(epoch, log=self.log_validation_phase_loss)
            if self.is_early_stopping(validation_loss): self.early_stop_saving()
                
        self.training_end_saving()
        
                         
    def real_phase_train(self, real_data)->None:
        self.optimizer_D.zero_grad()
        # real_labels = torch.ones(real_data.size(0), 1).double()
        real_labels = self.generate_labels(self.are_smoothed_labels, 1, real_data.shape)
        
        outputs = self.discriminator(real_data)
        d_loss = self.criterion(outputs, real_labels)
        d_loss.backward()
        self.optimizer_D.step()
        return d_loss.item()
        
    def fake_phase_train(self, batch_size)->None:
        # Train discriminator on fake data
        self.optimizer_D.zero_grad()
        fake_data = self.generator(batch_size).detach()  # Detach to avoid generator gradients
        # fake_labels = torch.zeros(fake_data.size(0), 1).double()
        fake_labels = self.generate_labels(self.are_smoothed_labels, 0, fake_data.shape)
        outputs = self.discriminator(fake_data)
        d_loss = self.criterion(outputs, fake_labels)
        d_loss.backward()
        self.optimizer_D.step()
            
        # Train generator
        self.optimizer_G.zero_grad()
        fake_data = self.generator(batch_size)  # Recompute to create a new graph
        outputs = self.discriminator(fake_data)
        g_loss = self.criterion(outputs, self.generate_labels(self.are_smoothed_labels, 1, fake_data.shape))  # Generator wants to fool the discriminator
        g_loss.backward()
        self.optimizer_G.step()
        
        return fake_data, d_loss.item(), g_loss.item()
    
    def phase_validation(self, epoch, log = True)->None:
        self.optimizer_G.zero_grad()
        for real_data, real_leak_label, real_localization in self.validation_dataloader:
            real_data = real_data.reshape(real_data.size(0), -1, real_data.size(3))
            real_data = real_data.permute(0, 2, 1)
            fake_data = self.generator(real_data.shape[0])  # Recompute to create a new graph
            #fake_labels = torch.zeros(real_data.size(0), 1).double()
            fake_labels = self.generate_labels(self.are_smoothed_labels, 0, fake_data.shape)
            #real_labels = torch.ones(real_data.size(0), 1).double()
            real_labels = self.generate_labels(self.are_smoothed_labels, 1, real_data.shape)
            validation_input_data = torch.cat([real_data, fake_data])
            validation_input_labels = torch.cat([real_labels, fake_labels])
            
            validation_outputs = self.discriminator(validation_input_data)
            validation_g_loss = self.criterion(validation_outputs, validation_input_labels.double())/validation_input_data.shape[0] 
            if log:
                self.log_validation_loss(validation_g_loss, epoch)
            return validation_g_loss.item()
        
    def is_early_stopping(self, accuracy)->bool:
        if self.best_accuracy is None:
            self.best_accuracy = accuracy
            return False

        # Check if accuracy has improved by more than min_delta
        if abs(accuracy - 0.5) < self.min_delta:
            self.epochs_no_improve += 1
        else:
            self.epochs_no_improve = 0
            self.best_accuracy = accuracy

        # Trigger early stopping if no improvement for 'patience' epochs
        if self.epochs_no_improve >= self.patience:
            return True
        return False
    
    def generate_labels(self, is_smooth:bool, value:int, data_shape)->torch.Tensor:
        if is_smooth:
            return torch.clamp(torch.normal(mean= value, std=0.5, size=(data_shape[0], 1)).double(), min=0.0, max=1.0)
        else:
            if value == 0:
                return torch.zeros(data_shape[0], 1).double()
            elif value == 1:
                return torch.ones(data_shape[0, 1]).double()
            else:
                return torch.clamp(torch.normal(mean= value, std=0.0, size=(data_shape[0], 1)).double(), min=0.0, max=1.0)      
    
    def log_hyperparameters(self)->None:
        self.experiment.log_parameters({
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.number_of_epochs
        })
    
    def log_losses(self, epoch:int, batch:int, real_d_loss:float, fake_d_loss:float, g_loss:float)->None:
        print(f"Epoch/Batch:  {epoch}/{batch}")
        print(f"\treal dicriminator loss:\t{real_d_loss}")
        print(f"\tfake dicriminator loss:\t{fake_d_loss}")
        print(f"\tgenerator loss:\t{g_loss}")
        
        if self.experiment:  # Ensure the experiment is not None
            self.experiment.log_metric("real_d_loss", real_d_loss, step=(epoch*self.batch_size) + batch)
            self.experiment.log_metric("fake_d_loss", fake_d_loss, step=(epoch*self.batch_size) + batch)
            self.experiment.log_metric("g_loss", g_loss, step=(epoch*self.batch_size) + batch)
    
    def log_validation_loss(self, loss, epoch):
        print(f"Epoch:  {epoch}")
        print("f\tvalidation loss:\t{loss}")
        
        if self.experiment:  # Ensure the experiment is not None
            self.experiment.log_metric("validation_loss", loss, step=epoch)
            
    def log_data_as_plot(self, data: torch.Tensor, epoch:int, batch:int, name:str = "data")->None:
        name = f"{epoch}/{batch}_{name}"
        index = random.randint(0,data.shape[0]-1)
        random_element = data[index]
        plt.figure(figsize=(80, 8))
        for i in range(1):
            plt.plot(random_element[i].detach().numpy(), label=f'manometr{i+1}', linestyle='-')
        self.experiment.log_figure(figure_name= name, figure= plt)
        plt.close()
        
    def test_saving(self):
        print("Training finished - model not converged")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(self.generator.state_dict(), os.path.join(self.main_dir, f"GAN_generator_model_last_{timestamp}.pth"))
        torch.save(self.discriminator.state_dict(), os.path.join(self.main_dir, f"GAN_discriminator_model_{timestamp}.pth"))
        
    def early_stop_saving(self):
        print("Early stopping - model converged")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(self.generator.state_dict(), os.path.join(self.main_dir, f"GAN_generator_model_{timestamp}.pth"))
        torch.save(self.discriminator.state_dict(), os.path.join(self.main_dir, f"GAN_discriminator_model_{timestamp}.pth"))
        exit(0)
        
    def training_end_saving(self):
        print("Training finished - model not converged")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(self.generator.state_dict(), os.path.join(self.main_dir, f"GAN_generator_model_last_{timestamp}.pth"))
        torch.save(self.discriminator.state_dict(), os.path.join(self.main_dir, f"GAN_discriminator_model_{timestamp}.pth"))  
        

if __name__ == "__main__":
    generator = Generator()
    discriminator = Discriminator()
    
    experiment = Experiment(
        api_key="V6xW30HU42MtnnpSl6bsGODZ1",    # Replace 'your-api-key' with your actual Comet API key
        project_name="LeakDetection"
    )
    
    dataGenerator = SynteticDataGenerator(generator.double(), discriminator.double(), experiment)
    train_dataset_1, validation_dataset_1, test_dataset_1 = dataGenerator.get_dset('/home/mbak/LeakDetection/data/localization/v2_samples126_lenght22_typeLocalisation.npz')
    train_dataset_2, validation_dataset_2, test_dataset_2 = dataGenerator.get_dset('/home/mbak/LeakDetectionwithGit/LeakDetection/data/localization/v2_samples1000_lenght22_typeLocalisation.npz')
    train_dataset = train_dataset_1+train_dataset_2
    validation_dataset = validation_dataset_1 +validation_dataset_2
    test_dataset = test_dataset_1 + test_dataset_2
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=len(test_dataset), drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), drop_last=True, pin_memory=True)
    

    dataGenerator.train(train_loader, validation_loader, 50)
    
    