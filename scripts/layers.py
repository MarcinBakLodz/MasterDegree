import torch 
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.functional as F

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.encoder = nn.Sequential(
      utils.parametrizations.weight_norm(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5,1), stride=(3,1), padding = 0)), # Convolutional layer 1
      nn.BatchNorm2d(16),
      nn.ReLU(),
      utils.parametrizations.weight_norm(nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(5,1), stride=(3,1), padding = 0)), # Convolutional layer 1
      nn.BatchNorm2d(4),
      nn.ReLU()
      )

  def forward(self, x):
    return self.encoder(x)

class Encoder2(nn.Module):
  def __init__(self):
    super(Encoder2, self).__init__()
    self.encoder = nn.Sequential(
      utils.parametrizations.weight_norm(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5,1), stride=(3,1), padding = 0)), # Convolutional layer 1
      nn.BatchNorm2d(16),
      nn.ReLU(),
      utils.parametrizations.weight_norm(nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(5,1), stride=(3,1), padding = 0)), # Convolutional layer 1
      nn.BatchNorm2d(4),
      nn.ReLU()
      )

  def forward(self, x):
    encoded_x = self.encoder(x)
    trimmed_to_one_sec_x = encoded_x[:, :, 1:-1, :]
    return trimmed_to_one_sec_x

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    self.decoder = nn.Sequential(
      utils.parametrizations.weight_norm(nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=(6,1), stride=(3,1), padding = 0)), # Convolutional layer 1
      nn.BatchNorm2d(16),
      nn.ReLU(),
      utils.parametrizations.weight_norm(nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(6,1), stride=(3,1), padding = (1,0))), # Convolutional layer 1
      nn.BatchNorm2d(1),
      nn.ReLU()
      )

  def forward(self, x):
    return self.decoder(x)

class Classifier2(nn.Module):
  def __init__(self):
    super(Classifier2, self).__init__()
    self.classifier2 = nn.Sequential(
      utils.parametrizations.weight_norm(nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(100,4), stride=(50,1), padding = 0)), # Convolutional layer 1
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(16*3, 1),
      nn.Sigmoid()
      )

  def forward(self, x):
    return self.classifier2(x)

class Localizator(nn.Module):
  def __init__(self):
    super(Localizator, self).__init__()
    self.localizator = nn.Sequential(
      utils.parametrizations.weight_norm(nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(100,4), stride=(50,1), padding = 0)), # Convolutional layer 1
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(16*3, 1),
      nn.Sigmoid()
      )

  def forward(self, x):
    return self.localizator(x)
  
########### GAN #############
  
class Generator(nn.Module):
  "Class to generate 20 secconds data"
  def __init__(self, latent_data_lenght: int = 100, latent_data_channels:int = 1, result_sampling_frequency: int =100, result_lenght_in_sec: int = 22, result_channels: int = 4, debug :bool = False):
    super().__init__()
    self.result_lenght_in_sec: int = result_lenght_in_sec
    self.latent_data_lenght: int = latent_data_lenght
    self.latent_data_channels: int = latent_data_channels
    self.result_sampling_frequency: int = result_sampling_frequency
    self.result_channels: int = result_channels
    self.debug = debug
    
    self.ch1 = 4
    self.ch2 = 64
    self.ch3 = 160
    self.ch4 = 250
    
    self.tc1 = torch.nn.ConvTranspose1d(self.latent_data_channels, self.ch1*self.latent_data_channels, kernel_size=(9), stride=1)
    self.batchNorm1 = torch.nn.BatchNorm1d(self.ch1*self.latent_data_channels)
    self.tc2 = torch.nn.ConvTranspose1d(self.ch1*self.latent_data_channels, self.ch2*self.latent_data_channels, kernel_size=(31), stride=2)
    self.batchNorm2 = torch.nn.BatchNorm1d(self.ch2*self.latent_data_channels)
    self.tc3 = torch.nn.ConvTranspose1d(self.ch2*self.latent_data_channels, self.ch3*self.latent_data_channels, kernel_size=(31), stride=3)
    self.batchNorm3 = torch.nn.BatchNorm1d(self.ch3*self.latent_data_channels)
    self.tc4 = torch.nn.ConvTranspose2d(self.ch3*self.latent_data_channels, self.ch3*self.latent_data_channels, kernel_size=(49,4), stride=(3, 1))
    self.batchNorm4 = torch.nn.BatchNorm2d(num_features=self.ch3*self.latent_data_channels)
    self.c1 = torch.nn.Conv2d(self.ch3*latent_data_channels, result_channels, kernel_size=(121,4), stride=(1,1), padding= 0)
    self.batchNormc1 = torch.nn.BatchNorm2d(result_channels)
    # self.c2 = torch.nn.Conv1d(8*latent_data_channels, result_channels, kernel_size=121, stride=1) #na wszelki wielki
    self.leakyRealu = torch.nn.LeakyReLU()
    self.tahn = torch.nn.Tanh()
    self.dropout02 = torch.nn.Dropout2d(p=0.2)
    self.dropout05 = torch.nn.Dropout2d(p=0.5)
    
  def generate_random_sample_from_gaussian_distribution(self, batch_size:int)-> torch.Tensor:
    for param in self.parameters():
      if param.dtype == torch.float32:
        return torch.normal(mean= 1, std=0.5, size=(batch_size, self.latent_data_channels, self.latent_data_lenght))
      elif param.dtype == torch.float64:
        return torch.normal(mean= 1, std=0.5, size=(batch_size, self.latent_data_channels, self.latent_data_lenght)).double()
      else:
        raise ValueError("Incorrect dicriminator noise format")

    
  def forward(self, batch_size:int)->torch.Tensor:
    x0 = self.generate_random_sample_from_gaussian_distribution(batch_size)
    if self.debug: print("x0: ", x0.shape)
    x1 = self.dropout02(self.leakyRealu(self.batchNorm1(self.tc1(x0))))
    if self.debug: print("x1: ", x1.shape)
    x2 = self.dropout05(self.leakyRealu(self.batchNorm2(self.tc2(x1))))
    if self.debug: print("x2: ", x2.shape)
    x3 = self.dropout05(self.leakyRealu(self.batchNorm3(self.tc3(x2))))
    if self.debug: print("x3: ", x3.shape)
    x31 = x3.unsqueeze(-1)
    if self.debug: print("x31: ", x31.shape)
    x4 = self.dropout05(self.leakyRealu(self.batchNorm4(self.tc4(x31))))
    if self.debug: print("x4: ", x4.shape)
    x5 = self.leakyRealu(self.batchNormc1(self.c1(x4)))
    if self.debug: print("x5: ", x5.shape)
    x51 = x5.squeeze(-1)
    if self.debug: print("x51: ", x51.shape)
    # x6 = self.LeakyRealu(self.c2(x51))
    # print("x6: ", x6.shape)
    start_index = (x51.shape[2] - self.result_lenght_in_sec*self.result_sampling_frequency) // 2
    end_index = start_index + self.result_lenght_in_sec*self.result_sampling_frequency
    x52 = x51[:, :, start_index:end_index]
    if self.debug: print("x52: ", x52.shape)
    return x52
  
class Discriminator(nn.Module):
  def __init__(self, input_channels:int = 4, debug:bool = False):
    super().__init__()
    self.input_channels = input_channels
    self.debug = debug
    
    self.ch1 = 8
    self.ch2 = 16
    self.ch3 = 32
    self.ch4 = 32
    
    self.c1 = torch.nn.Conv1d(in_channels=self.input_channels, out_channels=self.ch1, kernel_size=21, stride= 2)
    self.batchNorm1 = torch.nn.BatchNorm1d(self.ch1)
    self.c2 = torch.nn.Conv1d(self.ch1, out_channels=self.ch2, kernel_size=21, stride= 2)
    self.batchNorm2 = torch.nn.BatchNorm1d(self.ch2)
    self.c3 = torch.nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=(21,1), stride= (2,1))
    self.batchNorm3 = torch.nn.BatchNorm2d(self.ch3)
    self.c4 = torch.nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=(21,1), stride= (2,1))
    self.batchNorm4 = torch.nn.BatchNorm2d(self.ch4)
    self.l1 = torch.nn.Linear(in_features=self.ch4*119, out_features=1)
    
    
    self.leakyReLU = torch.nn.LeakyReLU()
    self.sigmoid = torch.nn.Sigmoid()
    self.dropout = torch.nn.Dropout(0.2)
    
    
  def forward(self, x:torch.Tensor)->torch.Tensor:
    assert self.check_if_input_has_good_shape(x), "bad input shape, it should be batch_size, 4, 2200"
    x1 = self.dropout(self.leakyReLU(self.batchNorm1(self.c1(x))))
    if self.debug: print("x1: ", x1.shape)
    x2 = self.dropout(self.leakyReLU(self.batchNorm2(self.c2(x1))))
    if self.debug: print("x2: ", x2.shape)
    x21 = x2.unsqueeze(-1)
    if self.debug: print("x21: ", x21.shape)
    x3 = self.dropout(self.leakyReLU(self.batchNorm3(self.c3(x21))))
    if self.debug: print("x3: ", x3.shape)
    x4 = self.dropout(self.leakyReLU(self.batchNorm4(self.c4(x3))))
    if self.debug: print("x4: ", x4.shape)
    x41 = torch.flatten(x4.squeeze(-1), start_dim= 1)
    if self.debug: print("x41: ", x41.shape)
    x5 = self.sigmoid(self.l1(x41))
    if self.debug: print("x5: ", x5.shape)
    return x5
    
  def check_if_input_has_good_shape(self, x:torch.Tensor)->bool:
    return x.shape[1:] == torch.Size([4, 2200])
    
  
########### STACK GAN #############
    
class StackGANGenerator1(nn.Module):
  def __init__(self, noise_lenght: int = 100, noise_channels:int = 1, amount_of_labels:int = 2, latent_channels = 4, result_sampling_frequency: int =100, result_lenght_in_sec: int = 22, result_channels: int = 4, debug :bool = False):
    super().__init__()
    self.noise_lenght = noise_lenght
    self.noise_channels = noise_channels
    self.amount_of_labels = amount_of_labels
    self.result_channels = result_channels
    self.latent_channels = latent_channels
    self.debug = debug 
  
    self.label_parser = nn.Sequential(
      nn.Linear(self.amount_of_labels, self.amount_of_labels*self.noise_lenght),
      nn.ReLU(),
      nn.Unflatten(1, (self.amount_of_labels, self.noise_lenght))
    )
    
    self.step1 = nn.Sequential(
      nn.Conv1d(self.amount_of_labels + self.noise_channels, self.result_channels, 1),
      nn.ReLU(),
      nn.BatchNorm1d(self.result_channels),
      nn.Conv1d(self.latent_channels, self.latent_channels, 3, padding=1, padding_mode='replicate'),
      nn.ReLU(),
      nn.BatchNorm1d(self.result_channels)
    )
    
    self.step2 = nn.Sequential(
      nn.Conv1d(self.amount_of_labels + self.latent_channels, self.result_channels, 1),
      nn.ReLU(),
      nn.BatchNorm1d(self.result_channels),
      nn.Conv1d(self.latent_channels, self.latent_channels, 3, padding=1, padding_mode='replicate'),
      nn.ReLU(),
      nn.BatchNorm1d(self.result_channels)
    )
    
    
  def circular_padding_1d(self, x:torch.Tensor, padding_height:int = 2)->torch.Tensor:
    return torch.cat(
        (x[:, -padding_height:, :], x, x[:, :padding_height, :]), dim=1
    )

  def pars_label(self, distances:torch.Tensor, powers:torch.Tensor)->torch.Tensor:
    combined = torch.stack((distances, powers), dim = 1)
    label = self.label_parser(combined)
    return label
    
     
  def generate_random_sample_from_gaussian_distribution(self, batch_size:int)-> torch.Tensor:
    for param in self.parameters():
      if param.dtype == torch.float32:
        return torch.normal(mean= 1, std=0.5, size=(batch_size, self.noise_channels, self.noise_lenght))
      elif param.dtype == torch.float64:
        return torch.normal(mean= 1, std=0.5, size=(batch_size, self.noise_channels, self.noise_lenght)).double()
      else:
        raise ValueError("Incorrect dicriminator noise format")  
  
  def forward(self, batch_size, distance, powers):
    x0_n = self.generate_random_sample_from_gaussian_distribution(batch_size)
    x0_l = self.pars_label(distance, powers)
    x0 = torch.cat((x0_n, x0_l), dim= 1)
    if self.debug: print("x0:", x0.shape)
    
    x1_n = self.step1(x0)
    if self.debug: print("x1_n: ", x1_n.shape)
    if self.debug: print("x0_l:", x0_l.shape)
    if self.debug: print("---------")
    x1 = torch.cat((x1_n, x0_l), dim= 1)
    if self.debug: print("x1: ", x1.shape)
    x1 = F.interpolate(x1, size=(2 * x1.shape[2],), mode='linear', align_corners=False)
    if self.debug: print("x1_e: ", x1.shape)
    x2 = self.step2(x1)
    if self.debug: print("x2: ", x2.shape)
    
    return x2
    
class StackGANCritic1(nn.Module):
  def __init__(self, input_channels:int = 4, input_lenght:int = 220, debug:bool = False):
    super().__init__()
    self.input_channels = input_channels
    self.input_lenght = input_lenght
    self.debug = debug
    
    self.step1 = nn.Sequential(
      nn.Conv1d(self.input_channels, self.input_channels, (3), padding= 1),
      nn.ReLU(),
      nn.MaxPool1d(2, 2),
      nn.LayerNorm((self.input_channels, self.input_lenght//2)),
    )
    
    self.step2 = nn.Sequential(
      nn.Conv1d(2*self.input_channels, self.input_channels, (3), padding= 1),
      nn.ReLU(),
      nn.MaxPool1d(2, 2),
      nn.LayerNorm((self.input_channels, self.input_lenght//4)),
    )
    
    self.step3 = nn.Sequential(
      nn.Flatten(),
      nn.Linear(self.input_channels*self.input_lenght//4, 1),
      nn.ReLU()
    )
    
  def forward(self, x:torch.Tensor)->torch.Tensor:
    x1 = self.step1(x)
    if self.debug: print("x1: ", x1.shape)
    x_s = F.interpolate(x, size=(x1.shape[2],), mode='linear', align_corners=False)
    if self.debug: print("x_s: ", x_s.shape)
    x1_e = torch.cat((x1, x_s), dim= 1)
    x2 = self.step2(x1_e)
    if self.debug: print("x2: ", x2.shape)
    x3 = self.step3(x2)
    if self.debug: print("x3: ", x3.shape)
    return x3
       
  
if __name__ == "__main__":
  "place for all tests"
  generator = StackGANGenerator1(noise_lenght=110, noise_channels= 1, debug=True)
  distances = torch.rand(2)
  powers = torch.rand(2)
  distances = torch.zeros_like(distances)
  powers = torch.zeros_like(powers) 
  generated_data = generator(2, distances, powers)
  critic = StackGANCritic1(4)
  criticised_data = critic(generated_data)
