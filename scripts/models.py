import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class Trainer_v1_1:
    def __init__(self, length_of_data_in_seconds, encoder, decoder, classifier, experiment, patience=5, min_delta=0.001, device='cpu', reconstruction_weight = 0.5):
        self.length_of_data_in_seconds = length_of_data_in_seconds
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.classifier = classifier.to(device)
        self.encoder_criterion = nn.MSELoss()
        self.classification_criterion = nn.BCELoss()
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.classifier.parameters()), lr=0.0003)
        self.experiment = experiment
        self.patience = patience
        self.min_delta = min_delta
        self.device = device
        self.reconstruction_weight = reconstruction_weight

    def train(self, train_data, train_labels, val_data, val_labels, batch_size, epochs):
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        train_label_loader = DataLoader(train_labels, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        val_label_loader = DataLoader(val_labels, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train_loss = self.train_epoch(train_loader, train_label_loader, epoch)
            val_loss, all_preds, all_targets = self.validate(val_loader, val_label_loader, epoch)

            print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            if epoch % 5 == 0:
                self.experiment.log_metric("train_loss", train_loss,)
                self.experiment.log_metric("val_loss", val_loss)
                self.log_confusion_matrix(all_preds, all_targets, epoch)

            # Early stopping
            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model
                torch.save(self.encoder.state_dict(), 'best_encoder.pth')
                torch.save(self.decoder.state_dict(), 'best_decoder.pth')
                torch.save(self.classifier.state_dict(), 'best_classifier.pth')
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print("Early stopping")
                break

    def train_epoch(self, data_loader, label_loader, epoch):
        self.encoder.train()
        self.decoder.train()
        self.classifier.train()

        total_loss = 0
        batch_idx = 0
        for data, labels in zip(data_loader, label_loader):
            batch_idx +=1
            data, labels = data.to(self.device), labels.to(self.device)
            buffer = torch.zeros(data.shape[0], 4, self.length_of_data_in_seconds * 10, 4, device=self.device).float()
            end_of_previous_tensor = torch.zeros(data.shape[0], 1, 10, 4, device=self.device).float()
            last_end_of_last_tensor = torch.zeros(data.shape[0], 1, 10, 4, device=self.device).float()

            all_reconstruction_losses = []
            all_classification_losses = []

            for second in range(self.length_of_data_in_seconds):
                actual_data = data[:, second].float().unsqueeze(1)
                if second != self.length_of_data_in_seconds - 1:
                    next_data = data[:, second + 1, :10].float().unsqueeze(1)
                    input_data = torch.cat((end_of_previous_tensor, actual_data, next_data), dim=2)
                else:
                    input_data = torch.cat((end_of_previous_tensor, actual_data, last_end_of_last_tensor), dim=2)

                encoded_data = self.encoder(input_data)
                end_of_previous_tensor = actual_data[:, :, -10:]
                trimmed_to_one_second_data = encoded_data[:, :, 1:-1, :]
                decoded_data = self.decoder(trimmed_to_one_second_data)
                reconstruction_loss = self.encoder_criterion(decoded_data, actual_data)

                if epoch % 5 == 0:
                    self.experiment.log_metric("reconstruction_loss", reconstruction_loss, step=epoch * len(data_loader) + batch_idx)

                buffer_without_last_element = buffer[:, :, 10:, :]
                buffer = torch.cat((buffer_without_last_element, encoded_data), dim=2)
                classification_output = self.classifier(buffer)
                squeezed_classification_output = classification_output.squeeze(1)

                if second == self.length_of_data_in_seconds - 1:
                    classification_loss = self.classification_criterion(squeezed_classification_output, labels.float())
                    self.experiment.log_metric("classification_loss", classification_loss, step=epoch * len(data_loader)+ batch_idx)
                else:
                    classification_loss = torch.tensor(0.0, device=self.device)  # No classification loss for intermediate steps

                all_reconstruction_losses.append(reconstruction_loss)
                all_classification_losses.append(classification_loss)

            total_loss = reconstruction_loss * self.reconstruction_weight + classification_loss
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss.item() / len(data_loader)

    def validate(self, val_loader, val_label_loader, epoch):
        self.encoder.eval()
        self.decoder.eval()
        self.classifier.eval()

        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            batch_idx = 0
            for data, labels in zip(val_loader, val_label_loader):
                batch_idx +=1
                data, labels = data.to(self.device), labels.to(self.device)
                buffer = torch.zeros(data.shape[0], 4, self.length_of_data_in_seconds * 10, 4, device=self.device).float()
                end_of_previous_tensor = torch.zeros(data.shape[0], 1, 10, 4, device=self.device).float()
                last_end_of_last_tensor = torch.zeros(data.shape[0], 1, 10, 4, device=self.device).float()

                all_reconstruction_losses = []
                all_classification_losses = []

                for second in range(self.length_of_data_in_seconds):
                    actual_data = data[:, second].float().unsqueeze(1)
                    if second != self.length_of_data_in_seconds - 1:
                        next_data = data[:, second + 1, :10].float().unsqueeze(1)
                        input_data = torch.cat((end_of_previous_tensor, actual_data, next_data), dim=2)
                    else:
                        input_data = torch.cat((end_of_previous_tensor, actual_data, last_end_of_last_tensor), dim=2)

                    encoded_data = self.encoder(input_data)
                    end_of_previous_tensor = actual_data[:, :, -10:]
                    trimmed_to_one_second_data = encoded_data[:, :, 1:-1, :]
                    decoded_data = self.decoder(trimmed_to_one_second_data)
                    reconstruction_loss = self.encoder_criterion(decoded_data, actual_data)

                    buffer_without_last_element = buffer[:, :, 10:, :]
                    buffer = torch.cat((buffer_without_last_element, encoded_data), dim=2)
                    classification_output = self.classifier(buffer)
                    squeezed_classification_output = classification_output.squeeze(1)

                    if second == self.length_of_data_in_seconds - 1:
                        classification_loss = self.classification_criterion(squeezed_classification_output, labels.float())
                        all_preds.extend(squeezed_classification_output.cpu().numpy())
                        all_targets.extend(labels.cpu().numpy())
                    else:
                        classification_loss = torch.tensor(0.0, device=self.device)  # No classification loss for intermediate steps

                    all_reconstruction_losses.append(reconstruction_loss)
                    all_classification_losses.append(classification_loss)

                total_loss = sum(all_reconstruction_losses) + sum(all_classification_losses)

        return total_loss.item() / len(val_loader), all_preds, all_targets

    def test(self, data, labels, batch_size):
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        label_loader = DataLoader(labels, batch_size=batch_size, shuffle=True)

        for data, labels in zip(data_loader, label_loader):
            data, labels = data.to(self.device), labels.to(self.device)
            buffer = torch.zeros(data.shape[0], 4, self.length_of_data_in_seconds * 10, 4, device=self.device).float()
            end_of_previous_tensor = torch.zeros(data.shape[0], 1, 10, 4, device=self.device).float()
            last_end_of_last_tensor = torch.zeros(data.shape[0], 1, 10, 4, device=self.device).float()

    def log_confusion_matrix(self, preds, targets, epoch, ):
        preds = [1 if x > 0.5 else 0 for x in preds]
        cm = confusion_matrix(targets, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'Confusion Matrix at Epoch {epoch}')
        plt.show()
        self.experiment.log_confusion_matrix(targets, preds, title = f'Confusion Matrix at Epoch {epoch}')

class Trainer_v1_2:
    def __init__(self, length_of_data_in_seconds, encoder, decoder, classifier, experiment, patience=5, min_delta=0.001, device='cpu', reconstruction_weight = 0.5):
        self.length_of_data_in_seconds = length_of_data_in_seconds
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.classifier = classifier.to(device)
        self.encoder_criterion = nn.MSELoss()
        self.classification_criterion = nn.BCELoss()
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.classifier.parameters()), lr=0.0003)
        self.experiment = experiment
        self.patience = patience
        self.min_delta = min_delta
        self.device = device
        self.reconstruction_weight = reconstruction_weight

    def train(self, train_data, train_labels, val_data, val_labels, batch_size, epochs):
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        train_label_loader = DataLoader(train_labels, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        val_label_loader = DataLoader(val_labels, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train_loss = self.train_epoch(train_loader, train_label_loader, epoch)
            val_loss, all_preds, all_targets = self.validate(val_loader, val_label_loader, epoch)

            print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            if epoch % 5 == 0:
                self.experiment.log_metric("train_loss", train_loss,)
                self.experiment.log_metric("val_loss", val_loss)
                self.log_confusion_matrix(all_preds, all_targets, epoch)

            # Early stopping
            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model
                torch.save(self.encoder.state_dict(), 'best_encoder.pth')
                torch.save(self.decoder.state_dict(), 'best_decoder.pth')
                torch.save(self.classifier.state_dict(), 'best_classifier.pth')
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print("Early stopping")
                break

    def train_epoch(self, data_loader, label_loader, epoch):
        self.encoder.train()
        self.decoder.train()
        self.classifier.train()

        total_loss = 0
        batch_idx = 0
        for data, labels in zip(data_loader, label_loader):
            batch_idx +=1
            data, labels = data.to(self.device), labels.to(self.device)
            buffer = torch.zeros(data.shape[0], 4, self.length_of_data_in_seconds * 10, 4, device=self.device).float()
            end_of_previous_tensor = torch.zeros(data.shape[0], 1, 10, 4, device=self.device).float()
            last_end_of_last_tensor = torch.zeros(data.shape[0], 1, 10, 4, device=self.device).float()

            all_reconstruction_losses = []
            all_classification_losses = []
            total_loss_batch = 0
            one_cycle_reconstruction_loss = 0

            for second in range(self.length_of_data_in_seconds):
                actual_data = data[:, second].float().unsqueeze(1)
                if second != self.length_of_data_in_seconds - 1:
                    next_data = data[:, second + 1, :10].float().unsqueeze(1)
                    input_data = torch.cat((end_of_previous_tensor, actual_data, next_data), dim=2)
                else:
                    input_data = torch.cat((end_of_previous_tensor, actual_data, last_end_of_last_tensor), dim=2)

                encoded_data = self.encoder(input_data)
                end_of_previous_tensor = actual_data[:, :, -10:]
                trimmed_to_one_second_data = encoded_data[:, :, 1:-1, :]
                decoded_data = self.decoder(trimmed_to_one_second_data)
                reconstruction_loss = self.encoder_criterion(decoded_data, actual_data)
                one_cycle_reconstruction_loss += reconstruction_loss/self.length_of_data_in_seconds

                if epoch % 5 == 0:
                    self.experiment.log_metric("reconstruction_loss", reconstruction_loss, step=epoch * len(data_loader) + batch_idx)

                buffer_without_last_element = buffer[:, :, 10:, :]
                buffer = torch.cat((buffer_without_last_element, encoded_data), dim=2)
                classification_output = self.classifier(buffer)
                squeezed_classification_output = classification_output.squeeze(1)

                if second == self.length_of_data_in_seconds - 1:
                    classification_loss = self.classification_criterion(squeezed_classification_output, labels.float())
                    self.experiment.log_metric("classification_loss", classification_loss, step=epoch * len(data_loader)+ batch_idx)
                else:
                    classification_loss = torch.tensor(0.0, device=self.device)  # No classification loss for intermediate steps

                all_reconstruction_losses.append(reconstruction_loss)
                all_classification_losses.append(classification_loss)



            total_loss = one_cycle_reconstruction_loss * self.reconstruction_weight + classification_loss
            self.experiment.log_metric("total_cycle_reconstruction_loss", one_cycle_reconstruction_loss, step=epoch * len(data_loader)+ batch_idx)
            self.experiment.log_metric("total_cycle_classification_loss", classification_loss, step=epoch * len(data_loader)+ batch_idx)
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss.item() / len(data_loader)

    def validate(self, val_loader, val_label_loader, epoch):
        self.encoder.eval()
        self.decoder.eval()
        self.classifier.eval()

        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            batch_idx = 0
            for data, labels in zip(val_loader, val_label_loader):
                batch_idx +=1
                data, labels = data.to(self.device), labels.to(self.device)
                buffer = torch.zeros(data.shape[0], 4, self.length_of_data_in_seconds * 10, 4, device=self.device).float()
                end_of_previous_tensor = torch.zeros(data.shape[0], 1, 10, 4, device=self.device).float()
                last_end_of_last_tensor = torch.zeros(data.shape[0], 1, 10, 4, device=self.device).float()

                all_reconstruction_losses = []
                all_classification_losses = []

                for second in range(self.length_of_data_in_seconds):
                    actual_data = data[:, second].float().unsqueeze(1)
                    if second != self.length_of_data_in_seconds - 1:
                        next_data = data[:, second + 1, :10].float().unsqueeze(1)
                        input_data = torch.cat((end_of_previous_tensor, actual_data, next_data), dim=2)
                    else:
                        input_data = torch.cat((end_of_previous_tensor, actual_data, last_end_of_last_tensor), dim=2)

                    encoded_data = self.encoder(input_data)
                    end_of_previous_tensor = actual_data[:, :, -10:]
                    trimmed_to_one_second_data = encoded_data[:, :, 1:-1, :]
                    decoded_data = self.decoder(trimmed_to_one_second_data)
                    reconstruction_loss = self.encoder_criterion(decoded_data, actual_data)

                    buffer_without_last_element = buffer[:, :, 10:, :]
                    buffer = torch.cat((buffer_without_last_element, encoded_data), dim=2)
                    classification_output = self.classifier(buffer)
                    squeezed_classification_output = classification_output.squeeze(1)

                    if second == self.length_of_data_in_seconds - 1:
                        classification_loss = self.classification_criterion(squeezed_classification_output, labels.float())
                        all_preds.extend(squeezed_classification_output.cpu().numpy())
                        all_targets.extend(labels.cpu().numpy())
                    else:
                        classification_loss = torch.tensor(0.0, device=self.device)  # No classification loss for intermediate steps

                    all_reconstruction_losses.append(reconstruction_loss)
                    all_classification_losses.append(classification_loss)

                total_loss = sum(all_reconstruction_losses) + sum(all_classification_losses)

        return total_loss.item() / len(val_loader), all_preds, all_targets

    def test(self, data, labels, batch_size):
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        label_loader = DataLoader(labels, batch_size=batch_size, shuffle=True)

        for data, labels in zip(data_loader, label_loader):
            data, labels = data.to(self.device), labels.to(self.device)
            buffer = torch.zeros(data.shape[0], 4, self.length_of_data_in_seconds * 10, 4, device=self.device).float()
            end_of_previous_tensor = torch.zeros(data.shape[0], 1, 10, 4, device=self.device).float()
            last_end_of_last_tensor = torch.zeros(data.shape[0], 1, 10, 4, device=self.device).float()

    def log_confusion_matrix(self, preds, targets, epoch):
        preds = [1 if x > 0.5 else 0 for x in preds]
        cm = confusion_matrix(targets, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'Confusion Matrix at Epoch {epoch}')
        plt.show()
        self.experiment.log_confusion_matrix(targets, preds, title = f'Confusion Matrix at Epoch {epoch}')

class LocalizationTrainer_v1(nn.Module):
    def __init__(self, localizator, length_of_data_in_seconds, experiment, patience=5, min_delta=0.001, device='cpu', reconstruction_weight = 0.5):
        super(LocalizationTrainer_v1, self).__init__()

        self.localizator = localizator
        self.localization_criterion = nn.MSELoss()
        self.optimizer = optim.Adam(list(self.localizator.parameters()), lr=0.0003)
        self.length_of_data_in_seconds = length_of_data_in_seconds
        self.experiment = experiment
        self.patience = patience
        self.min_delta = min_delta
        self.device = device
        self.reconstruction_weight = reconstruction_weight

    def forward(self, x):
        return self.localizator(x)
    
    def train(self, dataset, path_to_encoder  = ""):
        if path_to_encoder != "":
            self.encoder = Encoder()
            encoder_path = 'path_to_encoder'
            self.encoder.load_state_dict(torch.load(encoder_path))
            self.encoder.eval()  # Przełączenie modelu w tryb ewaluacji
                

class SynteticDataGenerator(nn.Module):
    def __init__(self, generator, discriminator, learning_rate = 3e-3):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.learning_rate = learning_rate
        
        criterion = nn.BCELoss()
        optimizer_G = optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
    
    def train(num_of_epochs =10):
        for epoch in range(num_of_epochs):
            pass

if __name__ == "__main__":
    pass