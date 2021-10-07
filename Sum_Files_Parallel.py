# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 19:00:06 2021

@author: Gandalf
"""
from datetime import datetime
from multiprocessing import Process

def read_file(file):
  with open(file, "r") as inFile:
    for row in inFile:
      yield row

file_list = ["file1.txt", "file2.txt"]
file_generators = [read_file(path) for path in file_list]

file_list1 = ["file3.txt", "file4.txt"]
file_generators1 =[read_file(path) for path in file_list1]

def mult_process():
    with open("output_file2.txt", "w+") as outFile:   
        while True:
            try:
                outFile.write(f"{sum([int(next(gen)) for gen in file_generators + file_generators1])}\n")
            #outFile.write(f"{sum([int(next(gen)) for gen in file_generators1])}\n")
            except StopIteration:
                break
start = datetime.now()
if __name__ == '__main__':
    p = Process(target = mult_process, args = (10,))
    p.start()
    p.join()
end = datetime.now()
print("Start time is:", start)
print("End time is:", end)
print("Script took:", end-start)

