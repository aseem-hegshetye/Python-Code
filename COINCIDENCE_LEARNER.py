#CLASSIFYING CAR CONDITION
import sys
import operator
import math
import time
import numpy
#from numpy import *
#   "##" MEANS RECENTLY COMMENTED OUT FUNCTION
from TALK_support import *
from TALK_support2 import *
from CLASSIFICATION_SUPPORT import *
def main():
    
    connection_matrix_1=numpy.random.rand(CLASSIFICATION_SUPPORT.sdr_output_columns,CLASSIFICATION_SUPPORT.ascii_matrix_columns)
    connection_matrix_2=CLASSIFICATION_SUPPORT.Connection_Mat_2() # ITS NOT A REGULAR cm.
    connection_matrix_3=numpy.zeros((CLASSIFICATION_SUPPORT.sdr_output_columns,CLASSIFICATION_SUPPORT.sdr_output_columns))
    print ('connection_matrix_2',connection_matrix_2)
    
    prediction=[]
    #LEARNING !!
    my_str=['aseem','seema','sneha']
    my_str2=['haidr','abhij','milin'] # 2ND LIST OF INPUTS
    for i in range (0,(len(my_str))):
        cl_ob=CLASSIFICATION_SUPPORT()# EVERYTIME WE CREATE IT, ITS OLDER MEMORY/VALUES IS GONE
        cl_ob2=CLASSIFICATION_SUPPORT()# EVERYTIME WE CREATE IT, ITS OLDER MEMORY/VALUES IS GONE
        
        cl_ob.Input_to_Sdr(connection_matrix_1,my_str[i]) # cl_ob.sdr_matrix_1 has SDR OF THE SEQUENCE
        cl_ob2.Input_to_Sdr(connection_matrix_1,my_str2[i]) # cl_ob.sdr_matrix_1 has SDR OF THE SEQUENCE
        
        cl_ob.Value_Mat(connection_matrix_2)# cl_ob.output = OUTPUT !!!!TAKING A LONGGG TIME
        cl_ob2.Value_Mat(connection_matrix_2)# cl_ob2.output = OUTPUT !!!!TAKING A LONGGG TIME 

        #print('FIRST output is \n',cl_ob.output)
        print('SECOND output is \n',cl_ob2.output)

        
        connection_matrix_3=CLASSIFICATION_SUPPORT.CM33(connection_matrix_3,cl_ob.output,cl_ob2.output) #STRENGTHENING CONNECTIONS BETWEEN TWO OUTPUT PATTERNS
        
        ##index=cl_ob.Extract_Indexes_ofOnes ()
        #print(index)
        #classification_vec = input("Enter the Disease number please \n ")
        ##CLASSIFICATION_SUPPORT.Disease_Classifying(l1,l2,l3,l4,l5,l6,i+1,index)#SINCE VALUE IN PYTHON ARE PASSED BY REFERENCE ALWAYS.
                                                        #L1,L2,L3,L4,l5,l6 WILL AUTOMATICALLY BE UPDATED
        #CLASSIFICATION_SUPPORT.Disease_Classifying(l1,l2,l3,l4,l5,l6,classification_vec,index)  #REAL FUNCTION

    

    # TESTING !!

    while (True):
        my_str = input("Enter the sequence for TESTING please \n ")
        cl_ob=CLASSIFICATION_SUPPORT()# EVERYTIME WE CREATE IT, ITS OLDER MEMORY/VALUES IS GONE
        cl_ob.Input_to_Sdr(connection_matrix_1,my_str) # cl_ob.sdr_matrix_1 has SDR OF THE SEQUENCE
        
        cl_ob.Value_Mat(connection_matrix_2)# cl_ob.output = OUTPUT
        #print('output is \n',cl_ob.output)
        #index=cl_ob.Extract_Indexes_ofOnes ()
        #print('index of ones in new input' , index)

        #------ PREDICTING 2ND INPUT
        output=(cl_ob.output*connection_matrix_3).sum(axis=1) # one ascii row
        output=SENTENCE2.top_bit(output,CLASSIFICATION_SUPPORT.on_bits_sdr)

        print('final output is \n',output)
        








        
if __name__ == '__main__':
  main()
