# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 19:00:06 2021

@author: Gandalf
"""
from datetime import datetime
import threading

def read_file(file):
  with open(file, "r") as inFile:
    for row in inFile:
      yield row

file_list = ["file1.txt", "file2.txt", "file3.txt"]
file_generators = [read_file(path) for path in file_list]

def add_file():
    with open("output_file.txt", "w+") as outFile:   
        while True:
            try:
                outFile.write(f"{sum([int(next(gen)) for gen in file_generators])}\n")
            except StopIteration:
                break
add = add_file()
start = datetime.now() 
# Call function with source text file and line count
t1 = threading.Thread(target = add, args=(10,))
t1.start()
t1.join()
#write()                
end = datetime.now()
print('Start time is:',start)
print('End time is:',end)
print('Script took:',end-start, 'secs')