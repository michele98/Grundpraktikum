import csv
import numpy as np

def change(filepath, new_name, delim = ",", ignore_line = 1):
    old_list = []
    
    #creates a list with the entries of the old csv
    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter = delim)
        old_list = list(reader)

    new_list = [old_list[i] for i in range(ignore_line)]
    
    #for every row in the wrong list
    for j in range(ignore_line,len(old_list)):
        
        #converts everything to string
        new_row = [str(entry) for entry in old_list[j]]
        #
        new_row = [new_row[i].replace(",",".") for i in range(len(new_row))]
        #
        new_row = [float(new_row[i]) for i in range(len(new_row))]
        new_list.append(new_row)
    
    #new_arr = np.array(new_list)
    #print (new_arr)

    writer = csv.writer(open(new_name, "w"), delimiter = delim)
    writer.writerows(new_list)

if __name__ == "__main__":
    current_path = "../"
    filename = raw_input("Write old file name: ")
    new_name = raw_input("How do you call new file? ")
    #filename = "Kirch1_old.csv"
    #new_name = "bella.csv"
    change (current_path + filename, new_name)
