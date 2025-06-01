from comet_ml import Experiment  # Import Comet
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_preprocessing import *
from torch.utils.data import TensorDataset, DataLoader
from layers import Encoder2, Localizator

### SETTINGS
num_epochs = 3000
batch_size = 16
validation_split = 0.2
patience = 50
###

# Initialize Comet experiment
experiment = Experiment(
    api_key="V6xW30HU42MtnnpSl6bsGODZ1",
    project_name="leak-detection"
)

data = np.load("/home/mbak/LeakDetection/data/localization/v2_samples126_lenght20_typeLocalisation.npz", allow_pickle=True)
data = data["package_1"]
means, stds = calculate_means(data["matrix"])
data["matrix"] = normalize(data["matrix"], [0.42217916, 0.42217916, 0.42217916, 0.42217916], [0.03025766, 0.03025766, 0.03025766, 0.03025766])
dataset_size = len(data)
validation_size = int(dataset_size * validation_split)
train_size = dataset_size - validation_size

# Step 4: Split the dataset into train and validation
validation_dataset = data["matrix"][:validation_size]
train_dataset = data["matrix"][validation_size:]
validation_labels = data["label"][:validation_size]
train_labels = data["label"][validation_size:]
# Step 5: Create DataLoaders for each subset

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_data_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
train_label_loader = DataLoader(train_labels, batch_size=batch_size, shuffle=False)
val_label_loader = DataLoader(validation_labels, batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
    encoder = Encoder2()
    # Załaduj zapisane wagi modelu
    encoder_path = '/home/mbak/LeakDetection/models/best_encoder.pth'
    encoder.load_state_dict(torch.load(encoder_path))
    encoder.eval()  # Przełączenie modelu w tryb ewaluacji

    localizator = Localizator()
    optimizer = optim.Adam(localizator.parameters(), lr=3e-4)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
        
    for epoch in range(num_epochs):
        #train
        localizator.train()
        train_loss = 0.0
        for data, label in zip(train_data_loader, train_label_loader):
            buffer = torch.zeros(data.shape[0], 4,  200, 4).float()
            end_of_previous_tensor = torch.zeros(data.shape[0], 1, 10, 4).float()
            last_end_of_last_tensor = torch.zeros(data.shape[0], 1, 10, 4).float()
            

            encoded_data = []
            for second in range(data.size(1)):
                actual_data = data[:, second].float().unsqueeze(1)
                if second != 20 - 1:
                    next_data = data[:, second + 1, :10].float().unsqueeze(1)
                    input_data = torch.cat((end_of_previous_tensor, actual_data, next_data), dim=2)
                else:
                    input_data = torch.cat((end_of_previous_tensor, actual_data, last_end_of_last_tensor), dim=2)
                end_of_previous_tensor = actual_data[:, :, -10:]

                encoded_data.append(encoder(input_data))
                
            encoded_data = torch.cat(encoded_data, dim=2)

            
            # Forward pass
            outputs = localizator(encoded_data)
            loss = criterion(outputs, label.float())

            # Backward pass i optymalizacja
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        #evaluation
        localizator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, label in zip(val_data_loader, val_label_loader):
                buffer = torch.zeros(data.shape[0], 4,  200, 4).float()
                end_of_previous_tensor = torch.zeros(data.shape[0], 1, 10, 4).float()
                last_end_of_last_tensor = torch.zeros(data.shape[0], 1, 10, 4).float()
                

                encoded_data = []
                for second in range(data.size(1)):
                    actual_data = data[:, second].float().unsqueeze(1)
                    if second != 20 - 1:
                        next_data = data[:, second + 1, :10].float().unsqueeze(1)
                        input_data = torch.cat((end_of_previous_tensor, actual_data, next_data), dim=2)
                    else:
                        input_data = torch.cat((end_of_previous_tensor, actual_data, last_end_of_last_tensor), dim=2)
                    end_of_previous_tensor = actual_data[:, :, -10:]

                    encoded_data.append(encoder(input_data))
                    
                encoded_data = torch.cat(encoded_data, dim=2)

                
                # Forward pass
                outputs = localizator(encoded_data)
                loss = criterion(outputs, label.float())
                val_loss += loss.item()
                
    
        val_loss /= len(val_data_loader)
        train_loss /= len(train_data_loader)
        
        # Log the metrics to Comet
        experiment.log_metric("train_loss", train_loss, epoch=epoch+1)
        experiment.log_metric("val_loss", val_loss, epoch=epoch+1)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.10f}, Validation Loss: {val_loss:.10f}')

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(localizator.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                early_stop = True
                break

    if early_stop:
        torch.save(localizator.state_dict(), '/home/mbak/LeakDetection/models/best_localizator.pth')

    # End the experiment
    experiment.end()