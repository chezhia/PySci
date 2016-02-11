import csv
def fetch(csvfile):
    f = open(csvfile)
    alldata = [];               # List object to extract data from file.
    try:                        # Exception Handler
        reader = csv.reader(f)
        for row in reader:
            alldata.append(row[1:len(row)]);

    finally:
        f.close();
       
    return alldata 

def futuredates(csvfile):
    f = open(csvfile)
    alldata = [];               # List object to extract data from file.
    try:                        # Exception Handler
        reader = csv.reader(f)
        for j,row in enumerate(reader):
            if(j >= 49):
                alldata.append(row[0:1]);
                
    finally:
        f.close();
       
    return alldata 

#all = fetch("../data/stock_returns_base150.csv");
#print ([i for i in all[0:1]]);
def write_fcast(csvfile,data):
    with open(csvfile, 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',');
        a.writerows(data)