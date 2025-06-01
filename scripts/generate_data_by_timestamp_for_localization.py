"""
Author: Marcin Bąk
Description: This file generate dataset to train leak localization network in .npz (numpy zip) format
Args: name_of_csv_file, 
"""
### Settings
number_of_samples = 126
lenght_of_samples_in_sec = 22
visualize = True

input_data_filename = r"/home/mbak/LeakDetection/data/raw/2024-08-03_1.csv"
input_events_filename = r"/home/mbak/LeakDetection/data/raw/rejestr_zdarzen.txt"
result_path = r"/home/mbak/LeakDetectionwithGit/LeakDetection/data/localization"
###

import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime

def concatenate_numbers(num1, num2):
    # Convert num2 to a string and pad with zeros if necessary
    num2_str = str(num2).zfill(3)
    # Concatenate both numbers as strings
    result = str(num1) + num2_str
    return int(result)

def read_csv(filename):
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the first row
        for row in csv_reader:
            element = []
            element.append(concatenate_numbers(row[0], row[1]))
            int_preassure = [float(x) for x in row[2:6]]
            element.append(int_preassure)
            element.append(1 if any(item == "1.0" for item in row[6:11]) else 0)
            data.append(element)
        return data
    
def read_txt(file_path: str)-> np.array:
    data_list = []
    
    with open(file_path, 'r') as file:
        for line in file:
            time_str, value_str = line.strip().split('\t')
            value_float = float(value_str.replace(',', '.'))
            data_list.append([convert_readable_format_to_timestamp(time_str), value_float])
    
    data_array = np.array(data_list)
    return data_array

def extend_list(real_list1, real_list2, quantity_of_elements_to_extend, star_time):
    differences = []
    for manometr_read in range(4):
        differences.append((real_list2[1][manometr_read] - real_list1[1][manometr_read])/(quantity_of_elements_to_extend+1))

    extended = []
    for time_of_element in range(quantity_of_elements_to_extend):
        extended.append([star_time + 10*(time_of_element+1),[real_list1[1][0] + (1+time_of_element)*differences[0],
                                        real_list1[1][1] + (1+time_of_element)*differences[1],
                                        real_list1[1][2] + (1+time_of_element)*differences[2],
                                        real_list1[1][3] + (1+time_of_element)*differences[3]], 1 if (abs(real_list1[0] - real_list2[0])/quantity_of_elements_to_extend+1)*time_of_element >= 0.5 else 0 ])
    return extended

def add_elements_after_index(lst, index, new_elements):
    lst[index + 1:index + 1] = new_elements

def fullfil_data(data: list)-> list:
    "Function check if there are lack of information between miliseconds, if so it will fullfill it with line fitted to border values"
    i = 1
    while(i < len(data)-1):
        quantity_of_elements_to_fullfill = int(((data[i][0] - 10 - data[i-1][0])/10))
        if quantity_of_elements_to_fullfill > 0:
            extended_data = extend_list(data[i-1], data[i], quantity_of_elements_to_fullfill, data[i-1][0])
            data = data[:i] + extended_data +data[i:]
        i += 1
    return data

def convert_timestamp_to_readable_format(timestamp: int)-> str:
    timestamp_seconds = timestamp / 1000
    readable_date = datetime.datetime.fromtimestamp(timestamp_seconds)
    readable_format = readable_date.strftime('%Y-%m-%d %H:%M:%S')
    return readable_format

def convert_readable_format_to_timestamp(readable_date: str)-> int:
    readable_date = datetime.datetime.strptime(readable_date, '%Y-%m-%d %H:%M:%S')
    timestamp_seconds = readable_date.timestamp()
    timestamp_milliseconds = int(timestamp_seconds * 1000)
    return timestamp_milliseconds


def generate_data(data_list:list, timestamps: np.array, data_lenght_in_seconds):
    results = []
    for i, data in enumerate(data_list):
        if data[0] in timestamps[:, 0]:
            one_event = []
            for i2 in range(data_lenght_in_seconds):
                one_second = []
                for i3 in range(100):
                    one_second.append(data_list[i+i2*100+i3][1]) 
                one_event.append(one_second)

            for timestamp, meter in timestamps:
                if data[0] == timestamp:
                    print(meter)
                    results.append([np.array(one_event), meter/1500])
    results_datatype = [('matrix', np.float64, results[0][0].shape), ('label', np.float64), ('localization', np.float64)]
    results_array = np.zeros(len(results), dtype=results_datatype)
    for i in range(len(results)):
      results_array['matrix'][i] = results[i][0]
      results_array['label'][i] = 1
      results_array['localization'][i] = results[i][1]
      
    return results_array

def show_plot(data):
    i2 = 0
    for i in data:
        x = [j for j in range(len(np.reshape(i[0][:,:,0],[-1])))]
        for k in range(4):
            plt.plot(x, np.reshape(i[0][:,:,k],[-1]), label=f'manometr {k}')
        plt.title(f'Label {i[1]}')
        plt.xlabel('time [s]')
        plt.ylabel('Manometrs value [mA]')
        plt.legend()  # Add legend

        plt.show()
        plt.savefig(f'/home/mbak/LeakDetection/data/localization/plot_{i2}.png', dpi=300, bbox_inches='tight')  # Adjust file name and dpi as needed
        plt.clf()
        i2 += 1
        
if __name__ == "__main__":
    csv_data_list: list = read_csv(input_data_filename)
    csv_data_list = fullfil_data(csv_data_list)
    events_array: np.array = read_txt(input_events_filename)

    data_array = generate_data(csv_data_list, events_array, lenght_of_samples_in_sec)
    print(data_array["matrix"].shape)
    np.savez(f'{result_path}/v2_samples{data_array["matrix"].shape[0]}_lenght{lenght_of_samples_in_sec}_typeLocalisation.npz', package_1 =data_array)
    show_plot(data_array)
    print("done")