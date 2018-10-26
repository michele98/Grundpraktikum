import csv

def csv_to_list(filepath, delim = ","):
    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter = delim)
        my_list = list(reader)
    return my_list

def return_row(table, index = 0, name = None):
#see return_column for info
    row = []
    if name != None:
        for i in range(len(table)):
            if table[i][0] == name:
                index = i
    else:
        try:
            row.append(float(table[index][0]))
        except:
            row.append(table[index][0])
    
    for i in range(1, len(table[0])):
        try:
            row.append(table[index][i])
        except:
            print ("found non float element in csv")
    return row

def return_column(table, index = 0, name = None):
    column = []
    
    #if a name is given, it will set the column index as the one with that name
    if name != None:
        for i in range(len(table[0])):
            if table[0][i] == name:
                index = i

    #if no name is given, it returns the first column
    else:
        try: #tries to convert to float
            column.append(float(table[0][index]))
        except: #if it is not possible it just appends the item
            column.append(table[0][index])
    
    #starts from element 1 ignoring element 0 (which is usually a title)
    for i in range(1, len(table)):
        try:
            column.append(float(table[i][index]))
        except:
            print ("found non float element in csv")
    return column