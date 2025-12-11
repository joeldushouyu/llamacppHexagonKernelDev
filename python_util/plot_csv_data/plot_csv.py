import matplotlib.pyplot as plt
import numpy as np
import csv

import sys



def load_csv_from_file(filename:str):
    # Load data from CSV file
    
    # determine number of columns
    data = None
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        num_columns = len(headers)
        
        data = {header: [] for header in headers}
        for row in reader:
            for i in range(num_columns):
                data[headers[i]].append(float(row[i]))
    return data


def plot_csv_data(data:dict, title:str, xlabel:str, ylabel:str, output_filename:str):
    # Plot data from dictionary
    plt.figure(figsize=(10, 6))
    
    
    # It assume the first column is x-axis and rest are y-axis
    x_values = data[list(data.keys())[0]]
    for key in list(data.keys())[1:]:
        y_values = data[key]
        plt.plot(x_values, y_values, label=key)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_filename)
    # display the plot
    plt.show()
    #plt.close()
    
    
    
if __name__ == "__main__":
    # filename as first argument
    if len(sys.argv) < 2:
        print("Usage: python plotCsv.py <csv_filename>")
        sys.exit(1)
    
    
    csv_filename = sys.argv[1]
    plot_function_name = csv_filename.split('.')[0]
    data = load_csv_from_file(csv_filename)
    plot_csv_data(data, title=f"{plot_function_name} Approximation", xlabel="Input", ylabel="Output", output_filename=f"{plot_function_name}_plot.png")