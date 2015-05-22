import sys
import operator
import math
import time
import numpy
#from numpy import *

from TALK_support import *
from TALK_support2 import *
from CLASSIFICATION_SUPPORT import *
def main():
    
    connection_matrix_1=numpy.random.rand(CLASSIFICATION_SUPPORT.sdr_output_columns,CLASSIFICATION_SUPPORT.ascii_matrix_columns)
    
    connection_matrix_2=CLASSIFICATION_SUPPORT.Connection_Mat_2() # ITS NOT A REGULAR cm.
    while (True):
        cl_ob=CLASSIFICATION_SUPPORT()# EVERYTIME WE CREATE IT, ITS OLDER MEMORY/VALUES IS GONE
        cl_ob.Input_to_Sdr(connection_matrix_1) # cl_ob.sdr_matrix_1 has SDR OF THE SEQUENCE
        #print('sdr\n' , cl_ob.sdr_matrix_1)
        #print ('cm\n',connection_matrix_2)
        cl_ob.Value_Mat(connection_matrix_2)# cl_ob.output = OUTPUT
        print('output is \n',cl_ob.output)
        















        
if __name__ == '__main__':
  main()
