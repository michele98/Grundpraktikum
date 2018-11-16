import csv

def csv_to_list(filepath, delim = ","):
    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter = delim)
        my_list = list(reader)
    return my_list

def return_row(table, index = 0, title = None, title_index = 0):
#see return_column for info
    row = []
    if title != None:
        for i in range(len(table)):
            if table[i][title_index] == title:
                index = i
    else:
        try:
            row.append(float(table[index][title_index]))
        except:
            row.append(table[index][title_index])
    
    for i in range(1+title_index, len(table[title_index])):
        try:
            row.append(table[index][i])
        except:
            print ("found non float element in csv")
    return row

def return_column(table, index = 0, title = None, title_index = 0):
    #title_index is the row at which the title is
    column = []
    
    #if a title is given, it will set the column index as the one with that title
    if title != None:
        for i in range(len(table[title_index])):
            if table[title_index][i] == title:
                index = i

    #if no title is given, it returns the first column
    else:
        try: #tries to convert to float
            column.append(float(table[title_index][index]))
        except: #if it is not possible it just appends the item
            column.append(table[title_index][index])
    
    #starts from element 1 ignoring element 0 (which is usually a title)
    for i in range(1+title_index, len(table)):
        try:
            column.append(float(table[i][index]))
        except:
            print ("found non float element in csv")
    return column