import csv
import glob
import math
import scipy.stats

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.integrate import quad


class Bands:
    def __init__(self, index, lower_bound, upper_bound):
        self.index = index
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


def gaussian_bands(resolution):
    mean = 0
    std = 0.25
    x_min = -1.0
    x_max = 1.0
    x = np.linspace(x_min, x_max, 100)

    def normal_distribution_function(x):
        value = scipy.stats.norm.pdf(x, mean, std)
        return value

    total_area, err = quad(normal_distribution_function, x_min, x_max)
    total_area = round(total_area, 5)
    list_bands = list()
    upper_bound = 1
    index = 1
    for i in range(1, 2 * resolution + 1):
        x1 = (i - resolution - 1) / resolution
        x2 = (i - resolution) / resolution
        area, err = quad(normal_distribution_function, x1, x2)
        area = round(area, 5)
        length = round(2.0 * (area / total_area), 5)

        band = Bands(index, round(upper_bound - length, 5), round(upper_bound, 5))
        upper_bound = upper_bound - length
        list_bands.append(band)
        index += 1
    return list_bands


def quantization(df, bands):
    for band in bands:
        df.mask((df >= band.lower_bound) & (df < band.upper_bound), band.index, inplace=True)
    df.loc[:] = df.astype(int)
    return df


def write_quantized_data(data, directory, gesture, window_length, shift_length):
    file_name = gesture.split("\\")[-1].split(".")[0]
    new_file = directory + "/" + file_name + ".wrd"
    vector = list()
    with open(new_file, 'w', newline="") as x:
        for index, row in data.iterrows():
            for i in range(0, data.shape[1], shift_length):
                if i + window_length < data.shape[1]:
                    win = row[i:i + window_length].tolist()
                    pair = [file_name, index + 1, i]
                    vector.append(pair + win)
        csv.writer(x, delimiter=' ').writerows(vector)


def read_gestures_from_csv(all_files, directory, resolution, shift_length, window_length):
    print("Building Gaussian Bands...")
    bands = gaussian_bands(resolution)
    print('Reading data from the given folder and quantizing...')
    for filename in all_files:
        df = pd.read_csv(filename, header=None)
        column_names = [x for x in range(1, df.shape[1])]
        df = pd.DataFrame(df, columns=column_names)
        df_norm = df.subtract(df.min(axis=1), axis=0).multiply(2) \
            .divide(df.max(axis=1) - df.min(axis=1), axis=0).subtract(1).combine_first(df)
        quantized_data = quantization(df_norm, bands)
        write_quantized_data(quantized_data, directory, filename, window_length, shift_length)


def store_parameters(directory):
    with open("parameter.txt", 'w') as f:
        f.write(directory)
        f.close()


def task1(directory, window_length, shift_length, resolution):
    store_parameters(directory)
    all_files = glob.glob(directory + "/*.csv")
    read_gestures_from_csv(all_files, directory, resolution, shift_length, window_length)
    print("     ****Created .wrd files for all the gestures.****")


def get_all_words_from_directory(directory):
    words = list()
    all_files = glob.glob(directory + "/*.wrd")
    for filename in all_files:
        with open(filename, 'r') as f:
            for line in f:
                row = line.strip().split(' ')
                word = ' '.join(row[3:])
                if word not in words:
                    words.append(word)
    return dict.fromkeys(sorted(words), 0)


def create_word_dictionary(directory, all_words):
    all_files = glob.glob(directory + "/*.wrd")
    file_dict = dict()
    vectors = list()
    for file in all_files:
        parse_and_store_file_data(file_dict, file, all_words, vectors)
    df = pd.DataFrame(vectors)
    return file_dict, df


def parse_and_store_file_data(file_dict, file, all_words, vectors):
    file_name = file.split("\\")[-1].split(".")[0]
    with open(file, 'r') as f:
        sensor_dict = dict()
        for line in f:
            row = line.strip().split(' ')
            word = ' '.join(row[3:])
            if row[1] not in sensor_dict.keys():
                sensor_dict[row[1]] = all_words.copy()
            if word in sensor_dict[row[1]].keys():
                sensor_dict[row[1]][word] += 1
        file_dict[file_name] = sensor_dict
        for key, value in sensor_dict.items():
            vector = dict()
            vector["file"] = file_name
            vector["sensor"] = int(key)
            vector.update(value)
            vectors.append(vector)


def calculations(directory, data_dict, data_df, all_words):
    total_gestures = len(data_dict)
    tf = all_words.copy()
    tf_idf = all_words.copy()
    tf_idf2 = all_words.copy()
    vectors = list()
    for file_name, sensor_dict in data_dict.items():
        vector = list()
        vector.append(file_name)
        tf_vector = list()
        tf_idf_vector = list()
        tf_idf2_vector = list()
        for sensor, words_dict in sensor_dict.items():
            total_words = sum(words_dict.values())

            compute_tf_idf = data_df.loc[data_df['sensor'] == int(sensor)]
            num_of_docs_with_gesture = compute_tf_idf.astype(bool).sum(axis=0)

            compute_tf_idf2 = data_df.loc[data_df['file'] == file_name]
            num_of_words_in_gesture = compute_tf_idf2.astype(bool).sum(axis=0)

            for word, count in words_dict.items():
                tf[word] = count / total_words

                d_idf = float(num_of_docs_with_gesture[word])
                tf_idf[word] = float(tf[word]) * (math.log10(total_gestures / d_idf)) if d_idf > 0.0 else 0.0

                d_idf2 = float(num_of_words_in_gesture[word])
                tf_idf2[word] = float(tf[word]) * (
                    math.log10(len(compute_tf_idf2.columns) / d_idf2)) if d_idf2 > 0.0 else 0.0

            tf_vector.append(convert_vector_to_string(tf))
            tf_idf_vector.append(convert_vector_to_string(tf_idf))
            tf_idf2_vector.append(convert_vector_to_string(tf_idf2))

            tf = dict.fromkeys(tf, 0)
            tf_idf = dict.fromkeys(tf_idf, 0)
            tf_idf2 = dict.fromkeys(tf_idf2, 0)

        vector.append(" ".join(tf_vector))
        vector.append(" ".join(tf_idf_vector))
        vector.append(" ".join(tf_idf2_vector))
        vectors.append(vector)

    with open(directory + '/vectors.txt', 'w', newline="") as f:
        csv.writer(f, delimiter=',').writerows(vectors)
        f.close()


def convert_vector_to_string(data):
    data_str = [str(val) for val in list(data.values())]
    return " ".join(data_str)


def task2(directory):
    all_words = get_all_words_from_directory(directory)
    print("Building all words dictionary")
    data_dict, data_df = create_word_dictionary(directory, all_words)
    print("Performing TF, TF-IDF, TF-IDF2 calculations")
    calculations(directory, data_dict, data_df, all_words)
    print("     ****Created vectors.txt file****")


def task3(file):
    directory = read_parameters()
    all_words = get_all_words_from_directory(directory)
    vectors_txt = directory + "\\vectors.txt"
    file_name = file.split("\\")[-1].split(".")[0]
    tf = list()
    tf_idf = list()
    tf_idf2 = list()
    with open(vectors_txt, 'r') as f:
        for line in f:
            vector = line.strip().split(",")
            if vector[0] == file_name:
                tf_row_str = vector[1].split(" ")
                tf_row = [float(i) for i in tf_row_str]
                total_words = int(len(tf_row_str) / 20)
                tf_with_sensor = np.reshape(tf_row, (20, total_words))
                tf = tf_with_sensor.tolist()

                tf_idf_row_str = vector[2].split(" ")
                tf_idf_row = [float(i) for i in tf_idf_row_str]
                tf__idf_with_sensor = np.reshape(tf_idf_row, (20, total_words))
                tf_idf = tf__idf_with_sensor.tolist()

                tf_idf2_row_str = vector[3].split(" ")
                tf_idf2_row = [float(i) for i in tf_idf2_row_str]
                tf__idf2_with_sensor = np.reshape(tf_idf2_row, (20, total_words))
                tf_idf2 = tf__idf2_with_sensor.tolist()
        f.close()

    fig, ax = plt.subplots(figsize=(11, 9))
    plt.xlabel('Time')
    plt.ylabel('Sensors')

    while True:
        print("Press 1 for TF")
        print("Press 2 for TF-IDF")
        print("Press 3 for TF-IDF2")
        show_heat_map = input("Press a number to display Heatmap. Press 0 to exit\n")
        heat_map = int(show_heat_map)
        if heat_map not in (1, 2, 3):
            break
        if heat_map == 1:
            sensor_vs_time = create_sensor_vs_time(directory, file_name, tf, all_words)
            sb.heatmap(sensor_vs_time, cmap="Greys")
            plt.title("Heatmap Presenting TF for " + file_name, loc='center')
            plt.xlabel('Time')
            plt.ylabel('Sensors')
            plt.show()
        if heat_map == 2:
            sensor_vs_time = create_sensor_vs_time(directory, file_name, tf_idf, all_words)
            sb.heatmap(sensor_vs_time, cmap="Greys")
            plt.title("HeatMap Presenting TF-IDF for " + file_name, loc='left')
            plt.xlabel('Time')
            plt.ylabel('Sensors')
            plt.show()
        if heat_map == 3:
            sensor_vs_time = create_sensor_vs_time(directory, file_name, tf_idf2, all_words)
            sb.heatmap(sensor_vs_time, cmap="Greys")
            plt.title("HeatMap Presenting TF-IDF2 for " + file_name, loc='center')
            plt.xlabel('Time')
            plt.ylabel('Sensors')
            plt.show()


def create_sensor_vs_time(directory, file_name, based_on, semantic):
    df = pd.DataFrame(based_on, columns=semantic.keys())
    df.index = df.index + 1
    sensor_word = df.to_dict()
    time_dict = dict()
    with open(directory + "\\" + file_name + ".wrd", 'r') as f:
        for line in f:
            row = line.strip().split(" ")
            if int(row[1]) not in time_dict.keys():
                time_dict[int(row[1])] = dict()
            word = ' '.join(row[3:])
            if int(row[2]) not in time_dict[int(row[1])].keys():
                time_dict[int(row[1])][int(row[2])] = sensor_word[word][int(row[1])]
    sensor_time_df = pd.DataFrame.from_dict(time_dict, orient='index')
    df_norm = sensor_time_df.subtract(sensor_time_df.min(axis=1), axis=0).multiply(255).divide(
        sensor_time_df.max(axis=1) - sensor_time_df.min(axis=1), axis=0).combine_first(sensor_time_df)
    return df_norm


# to read the directory information from paramenter.txt
def read_parameters():
    with open("parameter.txt", 'r') as f:
        param = f.read()
        f.close()
    return param


def task4(file):
    directory = read_parameters()
    vectors_txt = directory + "\\vectors.txt"
    vector_dict = dict()
    with open(vectors_txt, 'r') as f:
        for line in f:
            vector = line.strip().split(",")
            vector_dict[vector[0]] = dict()

            tf_row_str = vector[1].split(" ")
            tf = [float(i) for i in tf_row_str]
            vector_dict[vector[0]]['tf'] = tf

            tf_idf_row_str = vector[2].split(" ")
            tf_idf = [float(i) for i in tf_idf_row_str]
            vector_dict[vector[0]]['tf_idf'] = tf_idf

            tf_idf2_row_str = vector[3].split(" ")
            tf_idf2 = [float(i) for i in tf_idf2_row_str]
            vector_dict[vector[0]]['tf_idf2'] = tf_idf2

    file_name = file.split("\\")[-1].split(".")[0]

    while True:
        print("Press 1 for TF")
        print("Press 2 for TF-IDF")
        print("Press 3 for TF-IDF2")
        similar_with = input("Press a number to display similar gestures. Press 0 to exit")
        similar_wrt = int(similar_with)
        if similar_wrt not in (1, 2, 3):
            break
        if similar_wrt == 1:
            get_similar_files(file_name, vector_dict, 'tf')
        if similar_wrt == 2:
            get_similar_files(file_name, vector_dict, 'tf_idf')
        if similar_wrt == 3:
            get_similar_files(file_name, vector_dict, 'tf_idf2')


def get_similar_files(file_name, vector_dict, based_on):
    results = dict()
    print("Following are the top 10 similar gesture files from the database for gesture file ", file_name, " based on ", based_on.upper())
    for key, file_value in vector_dict.items():
        if key != file_name:
            results[key] = euclidean_distance(file_value[based_on], vector_dict[file_name][based_on])
    list_results = sorted(results.items(), key=lambda x: x[1])
    for elem in list_results[:10]:
        print("     File Name: ", elem[0])


def euclidean_distance(x, y):
    return math.sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))


def main():
    while True:
        print("########## Phase 1 ##########")
        print("Task 1: Perform Normalization and Quantization")
        print("Task 2: Build Gesture Vectors")
        print("Task 3: Construct a Heatmap for a gesture")
        print("Task 4: Display 10 most similar gestures from the Database")
        task_input = input("What Task  do you want to perform: (enter 0 to exit)\n")
        task = int(task_input)
        if task not in (1, 2, 3, 4):
            print("Thank you. Have a good day!")
            break
        if task == 1:
            directory = input("Input Directory path of the gesture files:\n")
            window_length = int(input("Enter window length(w): \n"))
            shift_length = int(input("Enter shift length(s): \n"))
            resolution = int(input("Enter resolution (r): \n"))
            task1(directory, window_length, shift_length, resolution)
            print("########## Completed Task 1 ##########")
        if task == 2:
            directory = input("Input Directory path of the gesture files:\n")
            task2(directory)
            print("########## Completed Task 2 ##########")
        if task == 3:
            file = input("Input File with path to view the Heatmap:\n")
            task3(file)
            print("########## Completed Task 3 ##########")
        if task == 4:
            file = input("Input File with path for most similar gesture files:\n")
            task4(file)
            print("########## Completed Task 4 ##########")


if __name__ == '__main__':
    main()
