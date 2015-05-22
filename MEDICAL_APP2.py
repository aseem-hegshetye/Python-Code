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
    #ALL VARIABLES SHOULD BE SET HERE IN MAIN FUNCTION. CLASSES SHOULD REMAIN UNIVERSAL.
    
    
    #KP = KAGGLE_PLANKATON
    #KP_distinct_values=4;
    #KP_max_group_size=30; # SAME VARIABLE IN MATLAB. ACTUALLY max_group_size IS 2 TO 30 SO 28. BUT WE WILL KEEP 30. IT IS HOW MANY PIXELS CAN BE AT MAX IN A GROUP
    
##    KP_numOfOnes_in_group_length=4; #GIVES GOOD OVERLAP 12 AND 9 WILL HAVE 1 SIMILAR BIT.
##    sdr_output_columns=300
##    KP_percentage_of_on_bits_sdr=5
##    KP_on_bits_sdr=math.floor((KP_percentage_of_on_bits_sdr) * (KP_sdr_output_columns/100))
##    KP_max_noOf_groups_perQuad=20 # WE ASSUME THERE ARE 20 GROUPS / QUAD. WE DONT USE THIS ANYWHERE YET.BUT THIS DECIDES NUMBER OF CONNECTIONS PER NEURON. EVERY NEURON WILL REPRESENT TWO PARAMETERS PER PIX (VALUE /\|- AND LENGTH)
##    KP_no_of_connections_per_Neuron=40 #MAXIMUN LENGTH OF SEQUENCE THAT CAN BE CLASSIFIED CORRECTLY. BECAUSE ONLY AFTER THESE MANY COUNTS WILL THE RESPONSE GO IN OUTPUT
##                                    #IF INPUT IS 4 SEQ LONG, [THEN no_of_connections_per_Neuron-4] IS ALL WASTED.IT STAYS IN VALUE MATRIX UNTOUCHED. max_no_ofGroups=50 PER IMAGE IS IN MATLAB.
##                                    #KP_no_of_connections_per_Neuron IS PER QUADRANT. BUT THERE ARE TWO PARAMETERS PER INPUT GROUP.VALUE(/\-|) AND LENGTH.
##    KP_value_binary_Columns=KP_numOfOnes_in_values*KP_distinct_values #ITS NUMBER OF VALUE COLUMNS IN BINARY REPRESENTATION .4 DISTINCT VALUES. IF WE HAVE 2 ONES REPRESENTING A BINARY VALUE, THEN WE WOULD HAVE 8 BIT REPRESENTATION
##    KP_length_binary_Columns=KP_numOfOnes_in_group_length+KP_max_group_size# ITS NUMBER OF LENGTH COLUMNS IN BINARY REPRESENTATION. SINCE LENGTH IS AN ANALOG CATEGORY IT ITS OVERLAPING. sO WE DONT NEED A LOT OF BITS.OVERLAPPING REDUCES THE TOTAL LENTH OF BINARY REPRESENTATION NEEDED.
##    KP_numOfQuadrantReprsntatn=4 #ITS NUMBER OF QUADRANTS THAT MAKE FINAL IMAGE. IF WE CONSIDER LEARNING AN IMAGE BY ROTATING IT 4 TIMES THEN IT WILL BE 16
##    
##
##    cm1_value=numpy.random.rand(KP_sdr_output_columns,KP_value_binary_Columns) #VALUE MEANS WHAT CATEGORY THE PIXELS ARE IN ( / \ - |)
##    cm1_length=numpy.random.rand(KP_sdr_output_columns,KP_length_binary_Columns) # ITS LENGTH OF A GROUP OF PIXELS OF SAME VALUE
##    
##    
##    connection_matrix_2=CLASSIFICATION_SUPPORT.KP_Connection_Mat_2(KP_sdr_output_columns,KP_no_of_connections_per_Neuron) # ITS NOT A REGULAR cm. KP_sdr_output_columns HAS TO BE > KP_no_of_connections_per_Neuron
##    connection_matrix_2_img=CLASSIFICATION_SUPPORT.KP_Connection_Mat_2(KP_sdr_output_columns,KP_numOfQuadrantReprsntatn) # ITS NOT A REGULAR cm. SPECIALLY FOR CONVERTING QUADRANTS REPRESENTATION INTO ONE IMAGE REPRESENTATION.NUMBER OF QUADRANTS WE USE FOR AN IMAGE IS EQUAL TO THE NUMBER OF CONNECTIONS A NEURON SHOULD HAVE WHICH IS = COLUMNS OF THIS MATRIX
##    #print ('connection_matrix_2',connection_matrix_2)
    #############################################
    list_of_list=[]#BLANK LISTS TO STORE INDEXES OF ALL ONES BELONGING TO THIS CATEGORY.1st [] IS FOR NUMBER OF DISEASES.2ND [] IS FOR NUMBER OF TEST CASES IN THAT DISEASE
    NumberOfSymptoms=40
    no_of_connections_per_Neuron=NumberOfSymptoms #THATS THE TOTAL NUMBER OF INPUTS.. INPUTS WONT EVERY BE > OR < THEN TOTAL NUMBER OF SYMPTOMS. IF THE PATIENT DOESNT HAVE THAT SYMTOM IT WILL ATLEAST BE A "NO"
    #binLength=NumberOfSymptoms*2 # TOTAL LENGTH OF BINARY REPRESENTATION WILL BE TWICE NUMBER_OF_SYMPTOMS. COZ EVERY SYMPTOM WILL HAVE TWO OPTIONS 'YES' OR 'NO'
    numOfOnesInBin=2;
    sdr_output_columns=300
    percentage_of_on_bits_sdr=5
    on_bits_sdr=math.floor((percentage_of_on_bits_sdr) * (sdr_output_columns/100))
##    binaryColumns=(numOfOnesInBin*(NumberOfSymptoms*2)) #ITS NUMBER OF COLUMNS IN BINARY REPRESENTATION .4 DISTINCT VALUES. IF WE HAVE 2 ONES REPRESENTING A BINARY VALUE, THEN WE WOULD HAVE 8 BIT REPRESENTATION.EVERY SYMPTOM HAS 2 VALUES. (YES OR NO) SO THERE ARE (SYMPTOMS*2) DISTINCT VALUES.
    binaryColumns=(numOfOnesInBin*(NumberOfSymptoms)) #NOW WE JUST HAVE NumberOfSymptoms DISTINCT VALUES COZ IF SYMTOMS DOESNT EXIST ITS A PLAIN 0
    cm1_value=numpy.random.rand(sdr_output_columns,binaryColumns) # USED TO CONVERT BINARY TO SDR
    connection_matrix_2=CLASSIFICATION_SUPPORT.KP_Connection_Mat_2(sdr_output_columns,no_of_connections_per_Neuron) # ITS NOT A REGULAR cm. sdr_output_columns HAS TO BE > no_of_connections_per_Neuron
    l1=[]#BLANK LISTS TO STORE INDEXES OF ALL ONES BELONGING TO A DISEASE
    l2=[]
    l3=[]
    l4=[]
    l5=[]
    l6=[]
    l7=[]
    l8=[]
    l9=[]
    l10=[]
    l11=[]
    l12=[]
    l13=[]
    l14=[]
    l15=[]
    l16=[]
    l17=[]
    l18=[]
    l19=[]
    l20=[]
    l21=[]
    l22=[]
    l23=[]
    l24=[]
    l25=[]
    l26=[]
    l27=[]
    l28=[]
    l29=[]
    l30=[]
    l31=[]
    l32=[]
    l33=[]
    l34=[]
    l35=[]
    l36=[]
    l37=[]
    l38=[]
    l39=[]
    l40=[]
    l41=[]
    l42=[]
    l43=[]
    l44=[]
    l45=[]
    l46=[]
    l47=[]
    l48=[]
    l49=[]
    l50=[]
    l51=[]
    l52=[]
    l53=[]
    l54=[]
    l55=[]
    l56=[]
    l57=[]
    l58=[]
    l59=[]
    l60=[]
    l61=[]
    l62=[]
    l63=[]
    l64=[]
    l65=[]
    l66=[]
    l67=[]
    l68=[]
    l69=[]
    l70=[]
    l71=[]
    l72=[]
    l73=[]
    l74=[]
    l75=[]
    l76=[]
    l77=[]
    l78=[]
    l79=[]
    l80=[]
    l81=[]
    l82=[]
    l83=[]
    l84=[]
    l85=[]
    l86=[]
    l87=[]
    l88=[]
    l89=[]
    l90=[]
    l91=[]
    l92=[]
    l93=[]
    l94=[]
    l95=[]
    l96=[]
    l97=[]
    l98=[]
    l99=[]
    l100=[]
    l101=[]
    l102=[]
    l103=[]
    l104=[]
    l105=[]
    l106=[]
    l107=[]
    l108=[]
    l109=[]
    l110=[]
    l111=[]
    l112=[]
    l113=[]
    l114=[]
    l115=[]
    l116=[]
    l117=[]
    l118=[]
    l119=[]
    l120=[]
    l121=[]
    l122=[]
    l123=[]
    l124=[]


    #fin=open(r'C:\Users\ahegshetye\Downloads\clt\kag\FINAL_MAT.txt','r')
    count=0

    #LEARNING
    
    #ALL DISTINCT SYMPTOMS LIST .. SINCE INDEX IN PYTHON START FROM 0 WE WILL SUBSTRACT 1 FROM ALL INPUT SYMPTOMS
    input_symptoms=[[4,17,18,19],[1,17,18,6],[17,18,19,1,23,35,6],[17,18,19,1,23,35,22],[38,15,16,1,20,14,2,39,6],[17,18,19,1,2,39,6]]
    #4 17 18 19    1 17 18 6      38 15 16 1 20 14 2 39 6
    disease=[1,1,1,1,2,3] #DISEASE CLASSES 1 = DIABETES. DISEASE 2 = AIDS ETC.NO 0'S. WE WILL DO DISEASE-1  WHILE STORING SINCE PYTHON INDEX STARTS FROM 0
    
    for i in range (0,(len(input_symptoms))):
        rep=CLASSIFICATION_SUPPORT.Conv_inp_2Rep (input_symptoms[i],NumberOfSymptoms) #CONVERTS INPUT TO A REPRESENTATION. IF INPUT IS (1,2,3) AND NumberOfSymptoms=5 THEN REPRESENTATION=(1,2,3,0,0)
    #IF SYMPTOM EXIST IT HAS A NUMBER ELSE 0. SIMPLE AS SHIT. NO COMPLEX ODD EVEN BUSINESS FUCCKER
        
##        output=CLASSIFICATION_SUPPORT.Array_ofOddEvenNumbers(2,input_symptoms[i],NumberOfSymptoms) #RETURNS EVEN NUMBERS (NO'S) FOR ALL SYMPTOMS
##        #ABOVE OUTPUT WILL LOOK LIKE [1,3,6,8,10,12,14,16,18] FOR INPUT =[1,2].. NOTICE THAT THE FIRST TWO INDEXES ARE ODD 'YES'. OTHER VALUES ARE EVEN='NO'

        value1=[]
        for values in range(0,len(rep)):
            value_bin=(CLASSIFICATION_SUPPORT.Health_Converting_input_2binary(NumberOfSymptoms,numOfOnesInBin,rep[values],0)) #CONVERTING ALL VALUES EVEN OR ODD INTO  BINARY REPRESENTATION.IT SHOULD GIVE 0 IF INPUT IS 0
            value1.append(CLASSIFICATION_SUPPORT.Binary2Sdr(value_bin,cm1_value,on_bits_sdr))

        #print(len(value1))
        #print(value1[0])
        disease_rep=CLASSIFICATION_SUPPORT.Health_Value_Mat(connection_matrix_2,sdr_output_columns,on_bits_sdr,value1)
        #print(disease_rep)

        index=CLASSIFICATION_SUPPORT.KP_Extract_Indexes_ofOnes(on_bits_sdr,sdr_output_columns,disease_rep) # GIVES INDEXES OF ALL 1'S
        CLASSIFICATION_SUPPORT.KP_Classifying(l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18,l19,l20,l21,l22,l23,l24,l25,l26,l27,l28,l29,l30,l31,l32,l33,l34,l35,l36,l37,l38,l39,l40,l41,l42,l43,l44,l45,l46,l47,l48,l49,l50,l51,l52,l53,l54,l55,l56,l57,l58,l59,l60,l61,l62,l63,l64,l65,l66,l67,l68,l69,l70,l71,l72,l73,l74,l75,l76,l77,l78,l79,l80,l81,l82,l83,l84,l85,l86,l87,l88,l89,l90,l91,l92,l93,l94,l95,l96,l97,l98,l99,l100,l101,l102,l103,l104,l105,l106,l107,l108,l109,l110,l111,l112,l113,l114,l115,l116,l117,
l118,l119,l120,l121,l122,l123,l124,disease[i],index)
        
    CLASSIFICATION_SUPPORT.Converting_Lists2_ListOfList(l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18,l19,l20,l21,l22,l23,l24,l25,l26,l27,l28,l29,l30,l31,l32,l33,l34,l35,l36,l37,l38,l39,l40,l41,l42,l43,l44,l45,l46,l47,l48,l49,l50,l51,l52,l53,l54,l55,l56,l57,l58,l59,l60,l61,l62,l63,l64,l65,l66,l67,l68,l69,l70,l71,l72,l73,l74,l75,l76,l77,l78,l79,l80,l81,l82,l83,l84,l85,l86,l87,l88,l89,l90,l91,l92,l93,l94,l95,l96,l97,l98,l99,l100,l101,l102,l103,l104,l105,l106,l107,l108,l109,l110,l111,l112,l113,l114,l115,l116,l117,
l118,l119,l120,l121,l122,l123,l124,list_of_list)
    #print(list_of_list)

    #TESTING
    while (True):
        my_str = input("\n\n\n\n\n ENTER SYMPTOMS PLEASE \n ")
        l=my_str.split()
        op=list(map(int,l))
         
        rep=CLASSIFICATION_SUPPORT.Conv_inp_2Rep (op,NumberOfSymptoms)
        
        value1=[]
        for values in range(0,len(rep)):
            value_bin=(CLASSIFICATION_SUPPORT.Health_Converting_input_2binary(NumberOfSymptoms,numOfOnesInBin,rep[values],0)) #CONVERTING ALL VALUES EVEN OR ODD INTO  BINARY REPRESENTATION.IT SHOULD GIVE 0 IF INPUT IS 0
            value1.append(CLASSIFICATION_SUPPORT.Binary2Sdr(value_bin,cm1_value,on_bits_sdr))

        disease_rep=CLASSIFICATION_SUPPORT.Health_Value_Mat(connection_matrix_2,sdr_output_columns,on_bits_sdr,value1)
        index=CLASSIFICATION_SUPPORT.KP_Extract_Indexes_ofOnes(on_bits_sdr,sdr_output_columns,disease_rep) # GIVES INDEXES OF ALL 1'S

        l_max_true=CLASSIFICATION_SUPPORT.KP_Predicting_Class(list_of_list,index,on_bits_sdr) #ITS WORKING RIGHT IGNORING THE ORDER OF ONES


        
        
        print('DIABETES PROBABILITY=',(l_max_true[0]/on_bits_sdr))
        print('TB PROBABILITY=',(l_max_true[1]/on_bits_sdr))
        print('NEW DISEASE PROBABILITY=',(l_max_true[2]/on_bits_sdr))
        
        #list_of_list[disease[i]-1].append(index)
    
##    
##            sdr_matrix1=numpy.zeros((KP_no_of_connections_per_Neuron,KP_sdr_output_columns)) #SDR_MAT OF A QUADRANT 1
##            sdr_matrix2=numpy.zeros((KP_no_of_connections_per_Neuron,KP_sdr_output_columns)) #SDR_MAT OF A QUADRANT 2
##            sdr_matrix3=numpy.zeros((KP_no_of_connections_per_Neuron,KP_sdr_output_columns)) #SDR_MAT OF A QUADRANT 3
##            sdr_matrix4=numpy.zeros((KP_no_of_connections_per_Neuron,KP_sdr_output_columns)) #SDR_MAT OF A QUADRANT 4
##            
##            CLASSIFICATION_SUPPORT.KP_Conv_list_to_Matrix(value1,length1,sdr_matrix1,len(value1),len(length1)) #CONVERTING LISTS (VALUE AND LENGHT INTO SDR MATRIX AND ARRANGING VALUE IN FIRST 20 ROWS AND LENGTH IN LAST 20 ROWS FOR EVERY QUAD SDR MAT
##            CLASSIFICATION_SUPPORT.KP_Conv_list_to_Matrix(value2,length2,sdr_matrix2,len(value2),len(length2))# sdr_matrix2 IS AUTOMATICALLY CHANGED INSIDE BY THE FUNCTION. ITS PASS BY REFERENCE
##            CLASSIFICATION_SUPPORT.KP_Conv_list_to_Matrix(value3,length3,sdr_matrix3,len(value3),len(length3))
##            CLASSIFICATION_SUPPORT.KP_Conv_list_to_Matrix(value4,length4,sdr_matrix4,len(value4),len(length4))
##
##            
##            quad1_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2,KP_sdr_output_columns,sdr_matrix1,KP_on_bits_sdr,KP_no_of_connections_per_Neuron,KP_max_noOf_groups_perQuad,1)
##            quad2_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2,KP_sdr_output_columns,sdr_matrix2,KP_on_bits_sdr,KP_no_of_connections_per_Neuron,KP_max_noOf_groups_perQuad,1)
##            quad3_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2,KP_sdr_output_columns,sdr_matrix3,KP_on_bits_sdr,KP_no_of_connections_per_Neuron,KP_max_noOf_groups_perQuad,1)
##            quad4_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2,KP_sdr_output_columns,sdr_matrix4,KP_on_bits_sdr,KP_no_of_connections_per_Neuron,KP_max_noOf_groups_perQuad,1)
##
##            
##            
##            #CONVERTING 4 QUADRANT REPRESENTATIONS INTO 1 IMAGE REPRESENTATION IN ORDER OF QUAD[1 2 3 4]
##            sdr_matrix_final=numpy.zeros((KP_numOfQuadrantReprsntatn,KP_sdr_output_columns)) #FINAL IMAGE SDR MATRIX . SINCE WE HAVE 4 QUADRANTS NOW!!
##
##            #print(' sum(sdr_matrix_final) = ',sum(sdr_matrix_final))
##            CLASSIFICATION_SUPPORT.Put_QuadReprntatn_in_SDR(quad1_rep,quad2_rep,quad3_rep,quad4_rep,sdr_matrix_final) #sdr_mat HAS BEEN SET
##            IMG_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2_img,KP_sdr_output_columns,sdr_matrix_final,KP_on_bits_sdr,KP_numOfQuadrantReprsntatn,4,0) #HERE FINAL SDR MATRIX WILL HAVE 1 REP PER QUAD. SO TOTAL 4 ROWS OF INPUTS
##            if sum(IMG_rep)==0:
##                print(sdr_matrix_final)
##                print(quad1_rep)
##                print(quad2_rep)
##                print(quad3_rep)
##                print(quad4_rep)
##                print(value1,'\n',value2,'\n',value3,'\n',value4,'\n',length1,'\n',length2,'\n',length3,'\n',length4)
##                input('is someting wrong with image representation. why is 000000')
##            
##            #print(IMG_rep)
##            print('count of images',count) #COUNT OF IMAGES 30200 ..  TOTAL CLASSIFICATIONS (L's) 121 .. 1 TO 121
##            count=count+1
##
##            index=CLASSIFICATION_SUPPORT.KP_Extract_Indexes_ofOnes(KP_on_bits_sdr,KP_sdr_output_columns,IMG_rep) # GIVES INDEXES OF ALL 1'S
##            
##            CLASSIFICATION_SUPPORT.KP_Classifying(l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18,l19,l20,l21,l22,l23,l24,l25,l26,l27,l28,l29,l30,l31,l32,l33,l34,l35,l36,l37,l38,l39,l40,l41,l42,l43,l44,l45,l46,l47,l48,l49,l50,l51,l52,l53,l54,l55,l56,l57,l58,l59,l60,l61,l62,l63,l64,l65,l66,l67,l68,l69,l70,l71,l72,l73,l74,l75,l76,l77,l78,l79,l80,l81,l82,l83,l84,l85,l86,l87,l88,l89,l90,l91,l92,l93,l94,l95,l96,l97,l98,l99,l100,l101,l102,l103,l104,l105,l106,l107,l108,l109,l110,l111,l112,l113,l114,l115,l116,l117,
##l118,l119,l120,l121,l122,l123,l124,classification_number,index);
####            print('length of l1 = ' ,len(l1))
####            print('length of l2 = ' ,len(l2))
####            print('length of l3 = ' ,len(l3))
####            print('length of l4 = ' ,len(l4))
##
##        except ValueError:
##            print(' all files are done reading , just write them in memory now ');
##            #save=input('do u want to save new connections[1] or not [0]? \n')
##            #if save=='1': # SAVING ALL MATRICES
##            
##            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l1',l1)
##            
##
##            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\cm1_value',cm1_value)
##            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\cm1_length',cm1_length)
##            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\connection_matrix_2',connection_matrix_2)
##            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\connection_matrix_2_img',connection_matrix_2_img)
##            
##            print('All connections saved')
##    
##            break;
##        
##
##    # TESTING
##    input('STARTING TESTING \n\n')
##    fin2=open(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\Mean_Matrix_small_test\FINAL_MAT_small_test.txt','r')
##    count=0
##    while(True): #IT WILL CONTINUE TILL THE MEAN MAT TEXT FILE ENDS AND GIVES ERROR LOL.
##        try:
##            (value1,value2,value3,value4,length1,length2,length3,length4,classification_number)=CLASSIFICATION_SUPPORT.KP_Read_1Img_MeanMat(fin2,cm1_value,cm1_length,KP_on_bits_sdr,KP_numOfOnes_in_values,KP_numOfOnes_in_group_length,KP_max_group_size,KP_distinct_values)
##            
##            sdr_matrix1=numpy.zeros((KP_no_of_connections_per_Neuron,KP_sdr_output_columns)) #SDR_MAT OF A QUADRANT 1
##            sdr_matrix2=numpy.zeros((KP_no_of_connections_per_Neuron,KP_sdr_output_columns)) #SDR_MAT OF A QUADRANT 2
##            sdr_matrix3=numpy.zeros((KP_no_of_connections_per_Neuron,KP_sdr_output_columns)) #SDR_MAT OF A QUADRANT 3
##            sdr_matrix4=numpy.zeros((KP_no_of_connections_per_Neuron,KP_sdr_output_columns)) #SDR_MAT OF A QUADRANT 4
##            
##            CLASSIFICATION_SUPPORT.KP_Conv_list_to_Matrix(value1,length1,sdr_matrix1,len(value1),len(length1)) #CONVERTING LISTS (VALUE AND LENGHT INTO SDR MATRIX AND ARRANGING VALUE IN FIRST 20 ROWS AND LENGTH IN LAST 20 ROWS FOR EVERY QUAD SDR MAT
##            CLASSIFICATION_SUPPORT.KP_Conv_list_to_Matrix(value2,length2,sdr_matrix2,len(value2),len(length2))# sdr_matrix2 IS AUTOMATICALLY CHANGED INSIDE BY THE FUNCTION. ITS PASS BY REFERENCE
##            CLASSIFICATION_SUPPORT.KP_Conv_list_to_Matrix(value3,length3,sdr_matrix3,len(value3),len(length3))
##            CLASSIFICATION_SUPPORT.KP_Conv_list_to_Matrix(value4,length4,sdr_matrix4,len(value4),len(length4))
##
##            quad1_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2,KP_sdr_output_columns,sdr_matrix1,KP_on_bits_sdr,KP_no_of_connections_per_Neuron,KP_max_noOf_groups_perQuad,1)
##            quad2_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2,KP_sdr_output_columns,sdr_matrix2,KP_on_bits_sdr,KP_no_of_connections_per_Neuron,KP_max_noOf_groups_perQuad,1)
##            quad3_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2,KP_sdr_output_columns,sdr_matrix3,KP_on_bits_sdr,KP_no_of_connections_per_Neuron,KP_max_noOf_groups_perQuad,1)
##            quad4_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2,KP_sdr_output_columns,sdr_matrix4,KP_on_bits_sdr,KP_no_of_connections_per_Neuron,KP_max_noOf_groups_perQuad,1)
##
##            #CONVERTING 4 QUADRANT REPRESENTATIONS INTO 1 IMAGE REPRESENTATION IN ORDER OF QUAD[1 2 3 4]
##            sdr_matrix_final=numpy.zeros((KP_numOfQuadrantReprsntatn,KP_sdr_output_columns)) #FINAL IMAGE SDR MATRIX . SINCE WE HAVE 4 QUADRANTS NOW!!
##            CLASSIFICATION_SUPPORT.Put_QuadReprntatn_in_SDR(quad1_rep,quad2_rep,quad3_rep,quad4_rep,sdr_matrix_final) #sdr_mat HAS BEEN SET
##            IMG_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2_img,KP_sdr_output_columns,sdr_matrix_final,KP_on_bits_sdr,KP_numOfQuadrantReprsntatn,4,0) #HERE FINAL SDR MATRIX WILL HAVE 1 REP PER QUAD. SO TOTAL 4 ROWS OF INPUTS
##            if sum(IMG_rep)==0:
##                print(sdr_matrix_final)
##                print(quad1_rep)
##                print(quad2_rep)
##                print(quad3_rep)
##                print(quad4_rep)
##                print(value1,'\n',value2,'\n',value3,'\n',value4,'\n',length1,'\n',length2,'\n',length3,'\n',length4)
##                input('is someting wrong with image representation. why is 000000')
##            
##            print(IMG_rep)
##            print('count of images',count)
##            count=count+1
##
##            index=CLASSIFICATION_SUPPORT.KP_Extract_Indexes_ofOnes(KP_on_bits_sdr,KP_sdr_output_columns,IMG_rep) # GIVES INDEXES OF ALL 1'S
##            
##            
##            l_max_true=CLASSIFICATION_SUPPORT.Predicting_Class(l1,l2,l3,l4,index)
##            print('l_max_true = ',l_max_true)
##
##        except ValueError:
##            print(' All test images are over.. done!! ');
##            break;
##    
##    
##    
##    #classification_vec IS CLASS OF CLASSIFICATION...input_string_mat IS INPUT CONVERTED INTO STRING WORD TO FEED THE ALGORITHM
##    print(input_string_mat,'\n',classification_vec)
##    no_of_Classifications=4 #NUMBER OF DISTINCT CLASSIFICATIONS IN THE DATASET
##    l1=[]#BLANK LISTS TO STORE INDEXES OF ALL ONES BELONGING TO THIS CATEGORY
##    l2=[]
##    l3=[]
##    l4=[]
##    prediction=[]
##    #LEARNING !!
##    for i in range(0,len(input_string_mat)):
##        
##        cl_ob=CLASSIFICATION_SUPPORT()# EVERYTIME WE CREATE IT, ITS OLDER MEMORY/VALUES IS GONE
##        cl_ob.Input_to_Sdr(connection_matrix_1,input_string_mat[i]) # cl_ob.sdr_matrix_1 has SDR OF THE SEQUENCE
##        #print('INP CONVERTED TO SDR')
##        #print('GENERATING OUTPUT ')
##        
##        cl_ob.Value_Mat(connection_matrix_2)# cl_ob.output = OUTPUT !!!!TAKING A LONGGG TIME 
##        
##        #print('output is \n',cl_ob.output)
##        index=cl_ob.Extract_Indexes_ofOnes ()
##        #print(index)
##         
##        CLASSIFICATION_SUPPORT.Classifying(l1,l2,l3,l4,classification_vec[i],index)#SINCE VALUE IN PYTHON ARE PASSED BY REFERENCE ALWAYS.
##                                                        #L1,L2,L3,L4 WILL AUTOMATICALLY BE UPDATED
##
##    
##    #print (l1,l2,l3,l4)
##    #print(l2)
##
##    # TESTING !!
##
##    while (True):
##        my_str = input("Enter the sequence please \n ")
##        cl_ob=CLASSIFICATION_SUPPORT()# EVERYTIME WE CREATE IT, ITS OLDER MEMORY/VALUES IS GONE
##        cl_ob.Input_to_Sdr(connection_matrix_1,my_str) # cl_ob.sdr_matrix_1 has SDR OF THE SEQUENCE
##        
##        cl_ob.Value_Mat(connection_matrix_2)# cl_ob.output = OUTPUT
##        #print('output is \n',cl_ob.output)
##        index=cl_ob.Extract_Indexes_ofOnes ()
##        print('index of ones in new input' , index)
##        l_max_true=CLASSIFICATION_SUPPORT.Predicting_Class(l1,l2,l3,l4,index)
##        print(l_max_true)
     



        
if __name__ == '__main__':
  main()
