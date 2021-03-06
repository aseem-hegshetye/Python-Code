#CLASSIFYING CAR CONDITION
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
    print ('connection_matrix_2',connection_matrix_2)
    #(input_string_mat,classification_vec)=CLASSIFICATION_SUPPORT.Read_txt() #READING TEXT FILE INTO INPUT STRING AND CLASSIFICATION NUMBER MATRIX.YAY
    #classification_vec IS CLASS OF CLASSIFICATION...input_string_mat IS INPUT CONVERTED INTO STRING WORD TO FEED THE ALGORITHM
    #print(input_string_mat,'\n',classification_vec)
    no_of_Classifications=6 #NUMBER OF DISTINCT CLASSIFICATIONS IN THE DATASET
    l1=[]#BLANK LISTS TO STORE INDEXES OF ALL ONES BELONGING TO THIS CATEGORY
    l2=[]
    l3=[]
    l4=[]
    l5=[]
    l6=[]
    prediction=[]
    #LEARNING !!
    my_str=['bdfhjkmprsuxzadhjlnprtvxzbdfgik','acfhikmprtuwybdhjenprtvxzbdfhjl','adfgilnprtuxyachjlnprtvxzbdfhjl','bdfhilnprtvxzbdgikmoqtvxzbdfhjl','bdfhjlnprtvxzbdhjenprsuwzbdfhjl','bdfgjLnprsvxzbdhjLnprtvxyacehjl']
    for i in range (0,(len(my_str))):
        cl_ob=CLASSIFICATION_SUPPORT()# EVERYTIME WE CREATE IT, ITS OLDER MEMORY/VALUES IS GONE
        cl_ob.Input_to_Sdr(connection_matrix_1,my_str[i]) # cl_ob.sdr_matrix_1 has SDR OF THE SEQUENCE
        #print('INP CONVERTED TO SDR')
        #print('GENERATING OUTPUT ')
        
        cl_ob.Value_Mat(connection_matrix_2)# cl_ob.output = OUTPUT !!!!TAKING A LONGGG TIME 
        
        #print('output is \n',cl_ob.output)
        index=cl_ob.Extract_Indexes_ofOnes ()
        #print(index)
        #classification_vec = input("Enter the Disease number please \n ")
        CLASSIFICATION_SUPPORT.Disease_Classifying(l1,l2,l3,l4,l5,l6,i+1,index)#SINCE VALUE IN PYTHON ARE PASSED BY REFERENCE ALWAYS.
                                                        #L1,L2,L3,L4,l5,l6 WILL AUTOMATICALLY BE UPDATED
        #CLASSIFICATION_SUPPORT.Disease_Classifying(l1,l2,l3,l4,l5,l6,classification_vec,index)  #REAL FUNCTION

    
    #print (l1,l2,l3,l4)
    #print(l2)

    # TESTING !!

    while (True):
        my_str = input("Enter the sequence for TESTING please \n ")
        cl_ob=CLASSIFICATION_SUPPORT()# EVERYTIME WE CREATE IT, ITS OLDER MEMORY/VALUES IS GONE
        cl_ob.Input_to_Sdr(connection_matrix_1,my_str) # cl_ob.sdr_matrix_1 has SDR OF THE SEQUENCE
        
        cl_ob.Value_Mat(connection_matrix_2)# cl_ob.output = OUTPUT
        #print('output is \n',cl_ob.output)
        index=cl_ob.Extract_Indexes_ofOnes ()
        print('index of ones in new input' , index)
        print(l1)
        l_max_true=CLASSIFICATION_SUPPORT.Disease_Predicting_Class(l1,l2,l3,l4,l5,l6,index)
        print(l_max_true)








        
if __name__ == '__main__':
  main()
