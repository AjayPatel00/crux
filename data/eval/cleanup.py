import csv

with open('eval.csv','r') as raw, open('eval_clean.csv','w',newline='') as clean:
    reader = csv.reader(raw)
    writer = csv.writer(clean)
    for row in reader:
        if len(row) == 0:
            continue
        elif len(row) == 1:
            continue
        elif len(row) == 2:
            writer.writerow(row)
        elif len(row) == 3:
            writer.writerow([row[0],row[1]])
            writer.writerow([row[0],row[2]])
        else:
            print(row)
