# FUNCTION FILE


#READING CSV FILE IN PYTHON LIST
import csv
x=[]
f=open(r'C:\Users\ahegshetye\Downloads\clt\kag\restaurnt\train.csv','rt')
reader=csv.reader(f)
for row in reader:
    x.append(row)
