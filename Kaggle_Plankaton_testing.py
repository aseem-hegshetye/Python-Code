
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
    KP_distinct_values=4;
    KP_max_group_size=30; # SAME VARIABLE IN MATLAB. ACTUALLY max_group_size IS 2 TO 30 SO 28. BUT WE WILL KEEP 30. IT IS HOW MANY PIXELS CAN BE AT MAX IN A GROUP
    KP_numOfOnes_in_values=2;
    KP_numOfOnes_in_group_length=4; #GIVES GOOD OVERLAP 12 AND 9 WILL HAVE 1 SIMILAR BIT.
    KP_sdr_output_columns=300
    KP_percentage_of_on_bits_sdr=5
    KP_on_bits_sdr=math.floor((KP_percentage_of_on_bits_sdr) * (KP_sdr_output_columns/100))
    KP_max_noOf_groups_perQuad=20 # WE ASSUME THERE ARE 20 GROUPS / QUAD. WE DONT USE THIS ANYWHERE YET.BUT THIS DECIDES NUMBER OF CONNECTIONS PER NEURON. EVERY NEURON WILL REPRESENT TWO PARAMETERS PER PIX (VALUE /\|- AND LENGTH)
    KP_no_of_connections_per_Neuron=40 #MAXIMUN LENGTH OF SEQUENCE THAT CAN BE CLASSIFIED CORRECTLY. BECAUSE ONLY AFTER THESE MANY COUNTS WILL THE RESPONSE GO IN OUTPUT
                                    #IF INPUT IS 4 SEQ LONG, [THEN no_of_connections_per_Neuron-4] IS ALL WASTED.IT STAYS IN VALUE MATRIX UNTOUCHED. max_no_ofGroups=50 PER IMAGE IS IN MATLAB.
                                    #KP_no_of_connections_per_Neuron IS PER QUADRANT. BUT THERE ARE TWO PARAMETERS PER INPUT GROUP.VALUE(/\-|) AND LENGTH.
    KP_value_binary_Columns=KP_numOfOnes_in_values*KP_distinct_values #ITS NUMBER OF VALUE COLUMNS IN BINARY REPRESENTATION .4 DISTINCT VALUES. IF WE HAVE 2 ONES REPRESENTING A BINARY VALUE, THEN WE WOULD HAVE 8 BIT REPRESENTATION
    KP_length_binary_Columns=KP_numOfOnes_in_group_length+KP_max_group_size# ITS NUMBER OF LENGTH COLUMNS IN BINARY REPRESENTATION. SINCE LENGTH IS AN ANALOG CATEGORY IT ITS OVERLAPING. sO WE DONT NEED A LOT OF BITS.OVERLAPPING REDUCES THE TOTAL LENTH OF BINARY REPRESENTATION NEEDED.
    KP_numOfQuadrantReprsntatn=4 #ITS NUMBER OF QUADRANTS THAT MAKE FINAL IMAGE. IF WE CONSIDER LEARNING AN IMAGE BY ROTATING IT 4 TIMES THEN IT WILL BE 16
    KP_no_of_classes=121
    
    #cm1_value  cm1_length   connection_matrix_2    connection_matrix_2_img CANT BE CREATED NEWLY NOW.. THE ONES THAT WERE USED TO LEARN THE IMAGES SHOULD BE REUSED AGAIN TO RECOGNIZE THE IMAGES. 
    list_of_list=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Marine plankton\train\saved\list_of_list.npy')
                            #C:\Users\Aseem\Documents\KAGGLE\Marine plankton\_saved
                            #C:\Users\Aseem\Documents\KAGGLE\Marine plankton\train\_saved
    

    cm1_value=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Marine plankton\train\saved\cm1_value.npy')
    cm1_length=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Marine plankton\train\saved\cm1_length.npy')
    CM3_CLASSIFIER=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Marine plankton\CM3_CLASSIFIER.npy')
    connection_matrix_2=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Marine plankton\train\saved\connection_matrix_2.npy')
    connection_matrix_2_img=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Marine plankton\train\saved\connection_matrix_2_img.npy')
    
    numpy.set_printoptions(threshold=numpy.nan) #PRINTING EVERYTHING !! WITHOUT ...
    
    fin=open(r'C:\Users\Aseem\Documents\KAGGLE\Marine plankton\FINAL_MAT_TEST.txt','r')
        #C:\Users\ahegshetye\Downloads\clt\_kag
        #C:\Users\Aseem\Documents\KAGGLE\_Marine plankton
    count=0
    l_max_true=[]
    prob_ofClasses_forAll_Images=[] # EVERY CLASS PROBABILITY FOR EVERY IMAGE
    while(True): #IT WILL CONTINUE TILL THE MEAN MAT TEXT FILE ENDS AND GIVES ERROR LOL.
        try:

            #print('starting KP_Read_1Img_MeanMat')
            (value1,value2,value3,value4,length1,length2,length3,length4)=CLASSIFICATION_SUPPORT.KP_Read_1Img_MeanMat_TESTING(fin,cm1_value,cm1_length,KP_on_bits_sdr,KP_numOfOnes_in_values,KP_numOfOnes_in_group_length,KP_max_group_size,KP_distinct_values)
            #print('End KP_Read_1Img_MeanMat')
            sdr_matrix1=numpy.zeros((KP_no_of_connections_per_Neuron,KP_sdr_output_columns)) #SDR_MAT OF A QUADRANT 1
            sdr_matrix2=numpy.zeros((KP_no_of_connections_per_Neuron,KP_sdr_output_columns)) #SDR_MAT OF A QUADRANT 2
            sdr_matrix3=numpy.zeros((KP_no_of_connections_per_Neuron,KP_sdr_output_columns)) #SDR_MAT OF A QUADRANT 3
            sdr_matrix4=numpy.zeros((KP_no_of_connections_per_Neuron,KP_sdr_output_columns)) #SDR_MAT OF A QUADRANT 4

            CLASSIFICATION_SUPPORT.KP_Conv_list_to_Matrix(value1,length1,sdr_matrix1,len(value1),len(length1)) #CONVERTING LISTS (VALUE AND LENGHT INTO SDR MATRIX AND ARRANGING VALUE IN FIRST 20 ROWS AND LENGTH IN LAST 20 ROWS FOR EVERY QUAD SDR MAT
            CLASSIFICATION_SUPPORT.KP_Conv_list_to_Matrix(value2,length2,sdr_matrix2,len(value2),len(length2))# sdr_matrix2 IS AUTOMATICALLY CHANGED INSIDE BY THE FUNCTION. ITS PASS BY REFERENCE
            CLASSIFICATION_SUPPORT.KP_Conv_list_to_Matrix(value3,length3,sdr_matrix3,len(value3),len(length3))
            CLASSIFICATION_SUPPORT.KP_Conv_list_to_Matrix(value4,length4,sdr_matrix4,len(value4),len(length4))

            #print('starting KP_Value_Mat matrix function QUADRANT 1')
            quad1_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2,KP_sdr_output_columns,sdr_matrix1,KP_on_bits_sdr,KP_no_of_connections_per_Neuron,KP_max_noOf_groups_perQuad,1)
            
            #print('starting KP_Value_Mat matrix function QUADRANT 2')
            quad2_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2,KP_sdr_output_columns,sdr_matrix2,KP_on_bits_sdr,KP_no_of_connections_per_Neuron,KP_max_noOf_groups_perQuad,1)
            #print('starting KP_Value_Mat matrix function QUADRANT 3')
            quad3_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2,KP_sdr_output_columns,sdr_matrix3,KP_on_bits_sdr,KP_no_of_connections_per_Neuron,KP_max_noOf_groups_perQuad,1)
            #print('starting KP_Value_Mat matrix function QUADRANT 4')
            quad4_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2,KP_sdr_output_columns,sdr_matrix4,KP_on_bits_sdr,KP_no_of_connections_per_Neuron,KP_max_noOf_groups_perQuad,1)

            #CONVERTING 4 QUADRANT REPRESENTATIONS INTO 1 IMAGE REPRESENTATION IN ORDER OF QUAD[1 2 3 4]
            
            sdr_matrix_final=numpy.zeros((KP_numOfQuadrantReprsntatn,KP_sdr_output_columns)) #FINAL IMAGE SDR MATRIX . SINCE WE HAVE 4 QUADRANTS NOW!!
            #print('STARTING Put_QuadReprntatn_in_SDR')
            CLASSIFICATION_SUPPORT.Put_QuadReprntatn_in_SDR(quad1_rep,quad2_rep,quad3_rep,quad4_rep,sdr_matrix_final) #sdr_mat HAS BEEN SET
            #print('ENDING Put_QuadReprntatn_in_SDR')
            #print('starting value matrix function')
            IMG_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2_img,KP_sdr_output_columns,sdr_matrix_final,KP_on_bits_sdr,KP_numOfQuadrantReprsntatn,4,0) #HERE FINAL SDR MATRIX WILL HAVE 1 REP PER QUAD. SO TOTAL 4 ROWS OF INPUTS
            #print('Ending value matrix function')
            if sum(IMG_rep)==0:
                print(sdr_matrix_final)
                print(quad1_rep)
                print(quad2_rep)
                print(quad3_rep)
                print(quad4_rep)
                print(value1,'\n',value2,'\n',value3,'\n',value4,'\n',length1,'\n',length2,'\n',length3,'\n',length4)
                input('is someting wrong with image representation. why is 000000')

            #print(IMG_rep)
            
            count=count+1
            print(count)
            
            index=CLASSIFICATION_SUPPORT.KP_Extract_Indexes_ofOnes(KP_on_bits_sdr,KP_sdr_output_columns,IMG_rep) # GIVES INDEXES OF ALL 1'S
            l_max_true.append(CLASSIFICATION_SUPPORT.KP_Predicting_Class(list_of_list,index,KP_on_bits_sdr))
            
            #prob_ofClasses_perImage=CLASSIFICATION_SUPPORT.KP_Predicting_Class_2(IMG_rep,CM3_CLASSIFIER,KP_no_of_classes,KP_on_bits_sdr)
            #prob_ofClasses_forAll_Images.append(prob_ofClasses_perImage)
            
            #print(prob_ofClasses_forAll_Images)
            #numpy.save(r'C:\Users\Aseem\Documents\KAGGLE\Marine plankton\prob_ofClasses_forAll_Images',prob_ofClasses_forAll_Images)
            #break
            print('l_max_true = ',l_max_true)
            
            input(' press enter for next image' )
            
        except ValueError:
            
            numpy.save(r'C:\Users\Aseem\Documents\KAGGLE\Marine plankton\prob_ofClasses_forAll_Images',prob_ofClasses_forAll_Images)
            print(' All test images are over..l_max_true SAVED. GOOD MORNING !!  done!! ');
            
            break;


        
if __name__ == '__main__':
  main()
