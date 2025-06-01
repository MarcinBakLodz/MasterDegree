
### Settings
number_of_samples = 1000
lenght_of_samples_in_sec = 22
visualize = False
ratio = 0.5

input_data_filename = r"/home/mbak/LeakDetection/data/raw/2024-03-28_1.csv"
result_path = r"/home/mbak/LeakDetectionwithGit/LeakDetection/data/localization"
###

import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt

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
            element.append(row[6:12])
            data.append(element)
        return data

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

def fullfil_data(data)-> list:
    "Function check if there are lack of information between miliseconds, if so it will fullfill it with line fitted to border values"
    i = 1
    while(i < len(data)-1):
        quantity_of_elements_to_fullfill = int(((data[i][0] - 10 - data[i-1][0])/10))
        if quantity_of_elements_to_fullfill > 0:
            extended_data = extend_list(data[i-1], data[i], quantity_of_elements_to_fullfill, data[i-1][0])
            data = data[:i] + extended_data +data[i:]
        i += 1
    return data

def generate_data_set(data:list, request_number_of_samples:int, leak_to_nonleak_ratio:float, quantity_of_sec_in_one_sample:int, label_format:str = "binary")->np.array:
    """
    possible formats:
        binary - if summary of all labels is >0.5 than it is 1
        max - if there is at least one 1 it is 1
        min - if there is at least one 0 it is 0
        float - percentage of leak in whole package
    """
    results = []
    acqual_number_of_leak = 0
    acqual_number_of_nonleak = 0
    request_number_of_leak = math.ceil(request_number_of_samples*leak_to_nonleak_ratio)
    request_number_of_nonleak = request_number_of_samples - request_number_of_leak

    tmp_leak_time = []
    tmp_nonleak_time = []
    leak_time = []
    nonleak_time = []
    stability_threshold = int(quantity_of_sec_in_one_sample*1000)
    actual_unstability = 0
    if data[0][2] == 1:
        state = "leak"
    else:
        state = "nonleak"
    for i in data[1010:-quantity_of_sec_in_one_sample*1010]: #nie bierze danych z końca
        if state == "leak":
            if i[2] == 0:
                actual_unstability +=1
            else:
                actual_unstability = 0

            if actual_unstability >= stability_threshold:
                state = "nonleak"
                leak_time.extend(tmp_leak_time)
                tmp_leak_time = []

        if state == "nonleak":
            if i[2] == 1:
                state = "leak"
                actual_unstability = 0

        if state =="leak":
            tmp_leak_time.append(i[0])
        elif state == "nonleak":
            tmp_nonleak_time.append(i[0])

    if len(tmp_leak_time) != 0:
        leak_time.extend(tmp_leak_time)
    if len(tmp_nonleak_time) != 0:
        nonleak_time.extend(tmp_nonleak_time)



    print(f"leak time / nonleak time : {len(leak_time)} / {len(nonleak_time)}")

    #turn data list into dict to optimize finding elements
    data_dict = {int(sublist[0]): sublist[1:] for sublist in data}

    actual_number_of_samples = 0

    iteration = 0
    while actual_number_of_samples < request_number_of_samples:
        "Generate one sample"
        iteration +=1
        print(f"iteration {iteration}")
        time_based_label = -1
        if acqual_number_of_nonleak < request_number_of_leak:
            random_time= random.choice(leak_time)
            time_based_label = 0
            leak_time.remove(random_time)
        else:
            random_time= random.choice(nonleak_time)
            time_based_label = 1
            nonleak_time.remove(random_time)
        starting_second = round(random_time/1000)*1000 #choosing starting second

        sample_values = []
        sample_labels = []
        sample_localization = []
        for second in range(starting_second, starting_second+quantity_of_sec_in_one_sample*1000, 1000):
            "Gather all reads from one second"
            one_second_values = []
            one_second_labels = []
            one_second_localization = []
            one_second_psik = []
            for milisecond in range(0,1000, 10):
                one_second_values.append(data_dict.get(second+milisecond)[0])
                one_second_psik.append(data_dict.get(second+milisecond)[1])
                if data_dict.get(second+milisecond)[1] == 1:
                    if data_dict.get(second+milisecond)[2][0] == 1:
                        sample_labels.append(1)
                        sample_localization.append(1428.06/1500)
                    if data_dict.get(second+milisecond)[2][1] == 1:
                        sample_labels.append(1)
                        sample_localization.append(1200.81/1500)
                    if data_dict.get(second+milisecond)[2][2] == 1:
                        sample_labels.append(1)
                        sample_localization.append(805.01/1500)
                    if data_dict.get(second+milisecond)[2][3] == 1:
                        sample_labels.append(1)
                        sample_labels.append(597.51/1500)
                    if data_dict.get(second+milisecond)[2][4] == 1:
                        sample_labels.append(1)
                        sample_localization.append(186.91/1500)
                    if data_dict.get(second+milisecond)[2][5] == 1:
                        sample_labels.append(1)
                        sample_localization.append(3.71/1500)
                else:
                    sample_labels.append(0)
                    sample_localization.append(0)

            sample_values.append(np.array(one_second_values))

        "process label"
        label = -1
        localization = 0
        if label_format == "binary":
            mean = sum(sample_labels) / len(sample_labels)
            if mean >= 0.5:
                label = 1
            else:
                label = 0
        if label_format == "max":
            is_any_equal_to_1 = False
            for element in sample_labels:
                if element == 1:
                    is_any_equal_to_1 = True
                    break
            if is_any_equal_to_1:
                label = 1
            else:
                label = 0
        if label_format == "min":
            if any(value == 1 for value in sample_labels):
                label = 1
            else:
                label = 0
        if label_format == "time_based":
            label = time_based_label
        if label_format == "float":
            label = sum(sample_labels) / len(sample_labels)
        if label_format == "distance":
            is_any_not_equal_to_0 = False
            actual_label = 0
            actual_localization = 0
            for (element, element_localization) in zip(sample_labels, sample_localization):
                if element != 0:
                    is_any_not_equal_to_0 = True
                    actual_label = element
                    actual_localization = element_localization
                    break
            if is_any_not_equal_to_0:
                label = actual_label
                localization = actual_localization
            else:
                label = 0
                localization = 0
                

        print(f" label: {label}")
        if label_format != "distance":
            if (label >= 0.5 and acqual_number_of_leak < request_number_of_leak) or (label < 0.5 and acqual_number_of_nonleak < request_number_of_nonleak):
                print("sample:")
                print(np.array(sample_values).squeeze().shape)
                print("label")
                sample = [np.array(sample_values).squeeze(), label, localization]
                results.append(sample)
                actual_number_of_samples +=1
                if label >= 0.5:
                    acqual_number_of_leak +=1
                else:
                    acqual_number_of_nonleak += 1
        else:
            sample = [np.array(sample_values).squeeze(), label, localization]
            results.append(sample)
            actual_number_of_samples +=1
            
        print(f" number of leak / number of nonleak : {acqual_number_of_leak} / {acqual_number_of_nonleak}")

    print(results[0][0].shape)
    print(results[0][0].dtype)
    results_datatype = [('matrix', np.float64, results[0][0].shape), ('label', np.float64), ('localization', np.float64)]
    
    results_array = np.zeros(len(results), dtype=results_datatype)
    for i in range(len(results)):
      results_array['matrix'][i] = results[i][0]
      results_array['label'][i] = results[i][1]
      results_array['localization'][i] = results[i][2]
    return results_array

def show_plot(data):
    for i in data:
        x = [j for j in range(len(np.reshape(i[0][:,:,0],[-1])))]
        for k in range(4):
            plt.plot(x, np.reshape(i[0][:,:,k],[-1]), label=f'manometr {k}')
        plt.title(f'Label {i[1]}')
        plt.xlabel('time [s]')
        plt.ylabel('Manometrs value [mA]')
        plt.legend()  # Add legend

        plt.show()

def show_ffplot(data, duration):
    fft_leak_sets = []
    fft_non_leak_sets = []
    for i in data:
        x = [j for j in range(len(np.reshape(i[0][:,:,0],[-1])))]
        for k in range(4):
            y = np.reshape(i[0][:,:,k],[-1])
            num_samples = len(y)  # Number of data points
            sampling_rate = num_samples / duration  # Sampling rate in Hz
            time = np.linspace(0, duration, num_samples)  # Time vector

            ##pressure_readings = np.sin(2 * np.pi * 1 * time) + np.sin(2 * np.pi * 2 * time)  # Example pressure signal with frequencies 1 Hz and 2 Hz
            pressure_readings = y
            pressure_readings -= np.mean(pressure_readings)

            fft_pressure = np.fft.fft(pressure_readings)
            freqs = np.fft.fftfreq(num_samples, 1 / sampling_rate)  # Frequency vector
            if i[1] == 1:
                fft_leak_sets.append(np.abs(fft_pressure[:num_samples//2]))
            else:
                fft_non_leak_sets.append(np.abs(fft_pressure[:num_samples//2]))

            derivative = np.zeros_like(pressure_readings)
            derivative[1:-1] = (pressure_readings[2:] - pressure_readings[:-2]) / 2.0
            # Use forward difference for the first point
            derivative[0] = (pressure_readings[1] - pressure_readings[0])
            # Use backward difference for the last point
            derivative[-1] = (pressure_readings[-1] - pressure_readings[-2])


            plt.subplot(3, 1, 1)
            plt.plot(time, pressure_readings)
            plt.title(f'Raw data')
            plt.xlabel('Time (s)')
            plt.ylabel('Pressure')
            plt.grid()

            plt.subplot(3, 1, 2)
            plt.plot(freqs[:num_samples//2], np.abs(fft_pressure[:num_samples//2]))
            plt.title('Fourier Transform of Pressure Signal')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.grid()

            plt.subplot(3, 1, 3)
            plt.plot(time, derivative)
            plt.title(f'First order derivative')
            plt.xlabel('Time (s)')
            plt.ylabel('derivative of preassure ')
            plt.grid()

            plt.tight_layout()
            plt.show()

    mean_leak = np.mean(fft_leak_sets, axis=0)
    mean_non_leak = np.mean(fft_non_leak_sets, axis = 0)

    plt.subplot(2, 1, 1)
    plt.plot(freqs[:num_samples//2], mean_non_leak)
    plt.title(f'non leak')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(freqs[:num_samples//2], mean_leak)
    plt.title('leak')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    csv_data = read_csv(input_data_filename)
    csv_data.pop(3)
    csv_data = fullfil_data(csv_data)

    data_array = generate_data_set(csv_data, number_of_samples, ratio, lenght_of_samples_in_sec, "distance")
    if visualize:
      print(data_array.shape)
      show_ffplot(data_array, 20)
      show_plot(data_array)


    np.savez(f'{result_path}/v2_samples{number_of_samples}_lenght{lenght_of_samples_in_sec}_typeLocalisation.npz', package_1 =data_array)