import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from os import listdir
from os.path import isfile, join
import cv2

def main():
   inputDir = sys.argv[1]
   #https://stackoverflow.com/questions/33369832/read-multiple-images-on-a-folder-in-opencv-python
   onlyfiles = [ f for f in listdir(inputDir) if isfile(join(inputDir,f)) ]
   images = np.empty(len(onlyfiles), dtype=object)
   filenames = np.empty(len(onlyfiles), dtype=string)
   for n in range(0, len(onlyfiles)):
      images[n] = cv2.imread( join(inputDir,onlyfiles[n]) )

if __name__ == '__main__':
    main()