from comet_ml import Experiment
from layers import StackGANGenerator1, StackGANCritic1
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from datasets import LocalizationDataFormat
import random
from datetime import datetime
import os
import torch.nn.functional as F


class SynteticDataGenerator(nn.Module):
    def __init__(self, generator1:StackGANGenerator1, critic1:StackGANCritic1, experiment:Experiment, learning_rate:float = 3e-3, lambda_gp:float = 10):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment = experiment
        self.generator1 = generator1.to(self.device)
        self.critic1 = critic1.to(self.device)
        self.learning_rate=learning_rate
        self.is_test_saving:bool = True
        self.lambda_gp = lambda_gp
        self.main_dir:str = r"C:\Users\Marcin\Desktop\Studia\Praca_dyplomowa\Wersja_styczniowa\LeakDetection\models\StackGAN"
        self.experiment = None
        self.log = True
        self.are_smoothed_labels = True
        
        #early_stopping
        self.min_delta = 0.05
        self.patience = 5        
        self.best_accuracy = None
        self.epochs_no_improve:int = 0
        self.number_of_epochs:int = 0
    
    
    def train_stage_1(self, train_dataloader:DataLoader, validation_dataloader:DataLoader,  number_of_epochs:int =10, critic_repetition:int = 1):
        if self.is_test_saving: self.test_saving()
        if self.experiment: self.log_hyperparameters()
        optimizer_g = optim.Adam(self.generator1.parameters(), lr=self.learning_rate, betas=(0.5, 0.9))
        optimizer_c = optim.Adam(self.critic1.parameters(), lr=self.learning_rate, betas=(0.5, 0.9))
        
        number_of_batches = len(train_dataloader)
        for epoch in range(number_of_epochs):
            # train phase
            for batch, (real_data, real_leak_label, real_localization) in enumerate(train_dataloader):
                real_data = real_data.reshape(real_data.size(0), -1, real_data.size(3))
                real_data = real_data.permute(0, 2, 1)
                real_data = real_data.to(self.device)
                real_data = F.interpolate(real_data, size=(real_data.shape[-1]//10,), mode='linear', align_corners=False)
                real_leak_label = real_leak_label.to(self.device)
                real_localization = real_localization.to(self.device)
                
                # critic part
                for i in range(critic_repetition):
                    fake_data = self.generator1(real_leak_label.shape[0], real_leak_label, real_localization)

                    real_output = self.critic1(real_data)
                    fake_output = self.critic1(fake_data)
                  
                    c_loss = -(real_output.mean() - fake_output.mean())
                    gp = self.gradient_penalty(real_data, fake_data)
                    c_loss += self.lambda_gp * gp
                    
                    
                    optimizer_c.zero_grad()
                    c_loss.backward()
                    optimizer_c.step()
                
                # generator part
                fake_data = self.generator1(real_leak_label.shape[0], real_leak_label, real_localization)
                fake_output = self.critic1(fake_data)
                
                g_loss = -fake_output.mean()
                
                optimizer_g.zero_grad()
                g_loss.backward()
                optimizer_g.step()
            
            if self.log:    
                self.log_losses(epoch, batch, "train", c_loss, g_loss) 
            
            #validation phase
            optimizer_g.zero_grad()
            c_loss_batch = 0
            g_loss_batch = 0

            number_of_validation_batches = len(validation_dataloader)
            for real_data, real_leak_label, real_localization in validation_dataloader:
                real_data = real_data.reshape(real_data.size(0), -1, real_data.size(3))
                real_data = real_data.permute(0, 2, 1)
                real_data = F.interpolate(real_data, size=(real_data.shape[-1]//10,), mode='linear', align_corners=False)
                real_leak_label = real_leak_label.to(self.device)
                real_localization = real_localization.to(self.device)
                
                fake_data = self.generator1(real_leak_label.shape[0], real_leak_label, real_localization)
                fake_labels = self.generate_labels(self.are_smoothed_labels, 0, fake_data.shape)
                real_labels = self.generate_labels(self.are_smoothed_labels, 1, real_data.shape)

                real_output = self.critic1(real_data)
                fake_output = self.critic1(fake_data)
                
                c_loss = -(real_output.mean() - fake_output.mean())
                gp = self.gradient_penalty(real_data, fake_data)
                c_loss += self.lambda_gp * gp
                c_loss_batch += c_loss
                
                g_loss += -fake_output.mean()
                g_loss_batch += g_loss
                
            c_loss_batch /= number_of_validation_batches
            g_loss_batch /= number_of_validation_batches
            
            if self.log:
                print(f"Epoch [{epoch+1}/{number_of_epochs}] | Critic Loss: {c_loss_batch:.4f} | Generator Loss: {g_loss_batch:.4f}")
                self.log_losses(epoch, batch, "validation", c_loss, g_loss)     
                self.log_data_as_plot(real_data, epoch, batch, "real_data")        
                self.log_data_as_plot(fake_data, epoch, batch, "generated_data") 
                
            if self.is_early_stopping(c_loss_batch): 
                self.early_stop_saving(1)
        
        self.very_end_saving()

    def get_dset(self, root_dir, train_share=0.8):
        dset = LocalizationDataFormat(root_dir)
        train_size = int(train_share * len(dset))
        validation_size = 2*(len(dset) - train_size)//3
        test_size = len(dset) - validation_size - train_size
        return random_split(dataset=dset, lengths=[train_size, validation_size, test_size], generator=torch.Generator().manual_seed(42))  # fix the generator for reproducible results
    
    def log_losses(self, epoch:int, batch:int, name:str, c_loss:float, g_loss:float)->None:
        print(f"{name} Epoch/Batch:  {epoch}/{batch}")
        print(f"\tcritic loss:\t{c_loss}")
        print(f"\tgenerator loss:\t{g_loss}")
        
        if self.experiment:  # Ensure the experiment is not None
            self.experiment.log_metric(f"c_loss_{name}", c_loss, step=(epoch*self.batch_size) + batch)
            self.experiment.log_metric(f"g_loss_{name}", g_loss, step=(epoch*self.batch_size) + batch)
    
    def log_data_as_plot(self, data: torch.Tensor, epoch:int, batch:int, name:str = "data", show_plot_on_screen:bool = False)->None:
        name = f"{epoch}/{batch}_{name}"
        index = 1
        random_element = data[index]
        plt.figure(figsize=(80, 8))
        for i in range(1):
            plt.plot(random_element[i].detach().numpy(), label=f'manometr{i+1}', linestyle='-')
            if show_plot_on_screen: plt.show()
        if self.experiment: self.experiment.log_figure(figure_name= name, figure= plt)
        plt.close()

    def test_saving(self):      
        print("Training starting - model not converged")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(self.generator1.state_dict(), os.path.join(self.main_dir, f"GAN_generator_model_test_{timestamp}.pth"))
        torch.save(self.critic1.state_dict(), os.path.join(self.main_dir, f"GAN_critic_model_test_{timestamp}.pth"))
        
    def gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size(0)
        epsilon = torch.randn_like(real_data, device=self.device)
        interpolates = epsilon * real_data + (1 - epsilon) * fake_data
        interpolates.requires_grad_(True)

        critic_output = self.critic1(interpolates)
        grad_outputs = torch.ones_like(critic_output, device=self.device)
        gradients = torch.autograd.grad(
            outputs=critic_output,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        return penalty
    
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

    def is_early_stopping(self, accuracy, mode = "min")->bool:
        if self.best_accuracy is None:
            self.best_accuracy = accuracy
        elif ((mode == "min" and accuracy < self.best_accuracy - self.min_delta) or
              (mode == "max" and accuracy > self.best_accuracy + self.min_delta)):
            self.best_accuracy = accuracy
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
                
        if self.best_accuracy is None:
            self.best_accuracy = accuracy
            return False


    def early_stop_saving(self, stage):
        print(f"Early stopping: {stage} - model converged")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(self.generator1.state_dict(), os.path.join(self.main_dir, f"GAN_generator_1_model_{timestamp}.pth"))
        torch.save(self.critic1.state_dict(), os.path.join(self.main_dir, f"GAN_critic_1_model_{timestamp}.pth"))
        if stage == 2 or stage == 3:
            torch.save(self.generator2.state_dict(), os.path.join(self.main_dir, f"GAN_generator_2_model_{timestamp}.pth"))
            torch.save(self.critic2.state_dict(), os.path.join(self.main_dir, f"GAN_critic_2_model_{timestamp}.pth"))
        if stage == 3:
            torch.save(self.generator3.state_dict(), os.path.join(self.main_dir, f"GAN_generator_3_model_{timestamp}.pth"))
            torch.save(self.critic3.state_dict(), os.path.join(self.main_dir, f"GAN_critic_3_model_{timestamp}.pth"))
        exit(0)
    
    def very_end_saving(self):
        print(f"End of training - model not converged")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(self.generator1.state_dict(), os.path.join(self.main_dir, f"GAN_generator_1_model_end{timestamp}.pth"))
        torch.save(self.critic1.state_dict(), os.path.join(self.main_dir, f"GAN_critic_1_model_end{timestamp}.pth"))
        # torch.save(self.generator2.state_dict(), os.path.join(self.main_dir, f"GAN_generator_2_model_end{timestamp}.pth"))
        # torch.save(self.critic2.state_dict(), os.path.join(self.main_dir, f"GAN_critic_2_model_end{timestamp}.pth"))
        # torch.save(self.generator3.state_dict(), os.path.join(self.main_dir, f"GAN_generator_3_model_end{timestamp}.pth"))
        # torch.save(self.critic3.state_dict(), os.path.join(self.main_dir, f"GAN_critic_3_model_end{timestamp}.pth"))
        exit(0)
    
    def log_hyperparameters(self)->None:
        self.experiment.log_parameters({
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.number_of_epochs
        })

if __name__ == "__main__":
    alone_model_generator = StackGANGenerator1(noise_lenght=110, noise_channels= 1, debug=False).double()
    alone_model_critic = StackGANCritic1(4).double()
    experiment = Experiment(
        api_key="V6xW30HU42MtnnpSl6bsGODZ1",    # Replace 'your-api-key' with your actual Comet API key
        project_name="LeakDetection_StackGAN"
    )
    dataGenerator = SynteticDataGenerator(alone_model_generator, alone_model_critic, experiment)
    
    train_dataset_1, validation_dataset_1, test_dataset_1 = dataGenerator.get_dset('C:\\Users\\Marcin\\Desktop\\Studia\\Praca_dyplomowa\\Wersja_styczniowa\\LeakDetection\\data\\localization\\v2_samples126_lenght22_typeLocalisation.npz')
    train_dataset_2, validation_dataset_2, test_dataset_2 = dataGenerator.get_dset('C:\\Users\\Marcin\\Desktop\\Studia\\Praca_dyplomowa\\Wersja_styczniowa\\LeakDetection\\data\\localization\\v2_samples1000_lenght22_typeLocalisation.npz')
    train_dataset = train_dataset_1+train_dataset_2
    validation_dataset = validation_dataset_1 + validation_dataset_2
    test_dataset = test_dataset_1 + test_dataset_2
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=len(test_dataset), drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), drop_last=True, pin_memory=True)
    
    dataGenerator.train_stage_1(train_loader, validation_loader, 10)
    
        