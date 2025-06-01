import os
import torch
import numpy as np
import torchaudio
from tqdm import tqdm
import data_preprocessing

class LocalizationDataFormat(torch.utils.data.Dataset):
    def __init__(self, root_dir = "/home/mbak/LeakDetection/data/localization/v2_samples126_lenght20_typeLocalisation.npz"):
        self.root_dir = root_dir
        
        self.data = []
        self.label = []
        self.localization = []
         
        origin_data = np.load(self.root_dir, allow_pickle=True)
        origin_data = origin_data["package_1"]
        
        print(origin_data.dtype)
        
        means, stds = data_preprocessing.calculate_means(origin_data["matrix"])
        origin_data["matrix"] = data_preprocessing.normalize(origin_data["matrix"], means, stds)
        for element in origin_data:
            
            self.data.append(element["matrix"])
            self.label.append(element["label"])
            self.localization.append(element["localization"])
        
    def __getitem__(self, index):
        return self.data[index], self.label[index], self.localization[index]

    def __len__(self):
        return len(self.data)
    
    def print_all_unique_localizations(self):
        used_localizations = []
        for localization in self.localization:
            if not round(localization, 2)  in used_localizations:
                print(round(localization, 2))
                used_localizations.append(round(localization, 2))
            
        

if __name__ == "__main__":
    loc = LocalizationDataFormat("/home/mbak/LeakDetection/data/localization/v2_samples126_lenght22_typeLocalisation.npz")
    loc.print_all_unique_localizations()