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
    

    cm1_value=numpy.random.rand(KP_sdr_output_columns,KP_value_binary_Columns) #VALUE MEANS WHAT CATEGORY THE PIXELS ARE IN ( / \ - |)
    cm1_length=numpy.random.rand(KP_sdr_output_columns,KP_length_binary_Columns) # ITS LENGTH OF A GROUP OF PIXELS OF SAME VALUE
    
    
    connection_matrix_2=CLASSIFICATION_SUPPORT.KP_Connection_Mat_2(KP_sdr_output_columns,KP_no_of_connections_per_Neuron) # ITS NOT A REGULAR cm. KP_sdr_output_columns HAS TO BE > KP_no_of_connections_per_Neuron
    connection_matrix_2_img=CLASSIFICATION_SUPPORT.KP_Connection_Mat_2(KP_sdr_output_columns,KP_numOfQuadrantReprsntatn) # ITS NOT A REGULAR cm. SPECIALLY FOR CONVERTING QUADRANTS REPRESENTATION INTO ONE IMAGE REPRESENTATION.NUMBER OF QUADRANTS WE USE FOR AN IMAGE IS EQUAL TO THE NUMBER OF CONNECTIONS A NEURON SHOULD HAVE WHICH IS = COLUMNS OF THIS MATRIX
    #print ('connection_matrix_2',connection_matrix_2)
    l1=[]#BLANK LISTS TO STORE INDEXES OF ALL ONES BELONGING TO THIS CATEGORY
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

    fin=open(r'C:\Users\ahegshetye\Downloads\clt\kag\FINAL_MAT.txt','r')
    count=0

    #LEARNING
    
    while(True): #IT WILL CONTINUE TILL THE MEAN MAT TEXT FILE ENDS AND GIVES ERROR LOL.
        try:
            #print('next image')
            (value1,value2,value3,value4,length1,length2,length3,length4,classification_number)=CLASSIFICATION_SUPPORT.KP_Read_1Img_MeanMat(fin,cm1_value,cm1_length,KP_on_bits_sdr,KP_numOfOnes_in_values,KP_numOfOnes_in_group_length,KP_max_group_size,KP_distinct_values) # VALUE1 VALUE 2 ETX ARE ALL LISTS
##            print('sum(sum(value1))= \n')
##            print(sum(sum(value1)))
##            print('sum(sum(value2))= \n')
##            print(len(value2))
##            
##            print('sum(sum(value3))= \n',sum(sum(value3)))
##            print('sum(sum(value4))= \n',sum(sum(value4)))
            sdr_matrix1=numpy.zeros((KP_no_of_connections_per_Neuron,KP_sdr_output_columns)) #SDR_MAT OF A QUADRANT 1
            sdr_matrix2=numpy.zeros((KP_no_of_connections_per_Neuron,KP_sdr_output_columns)) #SDR_MAT OF A QUADRANT 2
            sdr_matrix3=numpy.zeros((KP_no_of_connections_per_Neuron,KP_sdr_output_columns)) #SDR_MAT OF A QUADRANT 3
            sdr_matrix4=numpy.zeros((KP_no_of_connections_per_Neuron,KP_sdr_output_columns)) #SDR_MAT OF A QUADRANT 4
            
            CLASSIFICATION_SUPPORT.KP_Conv_list_to_Matrix(value1,length1,sdr_matrix1,len(value1),len(length1)) #CONVERTING LISTS (VALUE AND LENGHT INTO SDR MATRIX AND ARRANGING VALUE IN FIRST 20 ROWS AND LENGTH IN LAST 20 ROWS FOR EVERY QUAD SDR MAT
            CLASSIFICATION_SUPPORT.KP_Conv_list_to_Matrix(value2,length2,sdr_matrix2,len(value2),len(length2))# sdr_matrix2 IS AUTOMATICALLY CHANGED INSIDE BY THE FUNCTION. ITS PASS BY REFERENCE
            CLASSIFICATION_SUPPORT.KP_Conv_list_to_Matrix(value3,length3,sdr_matrix3,len(value3),len(length3))
            CLASSIFICATION_SUPPORT.KP_Conv_list_to_Matrix(value4,length4,sdr_matrix4,len(value4),len(length4))

            
            quad1_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2,KP_sdr_output_columns,sdr_matrix1,KP_on_bits_sdr,KP_no_of_connections_per_Neuron,KP_max_noOf_groups_perQuad,1)
            quad2_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2,KP_sdr_output_columns,sdr_matrix2,KP_on_bits_sdr,KP_no_of_connections_per_Neuron,KP_max_noOf_groups_perQuad,1)
            quad3_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2,KP_sdr_output_columns,sdr_matrix3,KP_on_bits_sdr,KP_no_of_connections_per_Neuron,KP_max_noOf_groups_perQuad,1)
            quad4_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2,KP_sdr_output_columns,sdr_matrix4,KP_on_bits_sdr,KP_no_of_connections_per_Neuron,KP_max_noOf_groups_perQuad,1)

            
            
            #CONVERTING 4 QUADRANT REPRESENTATIONS INTO 1 IMAGE REPRESENTATION IN ORDER OF QUAD[1 2 3 4]
            sdr_matrix_final=numpy.zeros((KP_numOfQuadrantReprsntatn,KP_sdr_output_columns)) #FINAL IMAGE SDR MATRIX . SINCE WE HAVE 4 QUADRANTS NOW!!

            #print(' sum(sdr_matrix_final) = ',sum(sdr_matrix_final))
            CLASSIFICATION_SUPPORT.Put_QuadReprntatn_in_SDR(quad1_rep,quad2_rep,quad3_rep,quad4_rep,sdr_matrix_final) #sdr_mat HAS BEEN SET
            IMG_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2_img,KP_sdr_output_columns,sdr_matrix_final,KP_on_bits_sdr,KP_numOfQuadrantReprsntatn,4,0) #HERE FINAL SDR MATRIX WILL HAVE 1 REP PER QUAD. SO TOTAL 4 ROWS OF INPUTS
            if sum(IMG_rep)==0:
                print(sdr_matrix_final)
                print(quad1_rep)
                print(quad2_rep)
                print(quad3_rep)
                print(quad4_rep)
                print(value1,'\n',value2,'\n',value3,'\n',value4,'\n',length1,'\n',length2,'\n',length3,'\n',length4)
                input('is someting wrong with image representation. why is 000000')
            
            #print(IMG_rep)
            print('count of images',count) #COUNT OF IMAGES 30200 ..  TOTAL CLASSIFICATIONS (L's) 121 .. 1 TO 121
            count=count+1

            index=CLASSIFICATION_SUPPORT.KP_Extract_Indexes_ofOnes(KP_on_bits_sdr,KP_sdr_output_columns,IMG_rep) # GIVES INDEXES OF ALL 1'S
            
            CLASSIFICATION_SUPPORT.KP_Classifying(l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18,l19,l20,l21,l22,l23,l24,l25,l26,l27,l28,l29,l30,l31,l32,l33,l34,l35,l36,l37,l38,l39,l40,l41,l42,l43,l44,l45,l46,l47,l48,l49,l50,l51,l52,l53,l54,l55,l56,l57,l58,l59,l60,l61,l62,l63,l64,l65,l66,l67,l68,l69,l70,l71,l72,l73,l74,l75,l76,l77,l78,l79,l80,l81,l82,l83,l84,l85,l86,l87,l88,l89,l90,l91,l92,l93,l94,l95,l96,l97,l98,l99,l100,l101,l102,l103,l104,l105,l106,l107,l108,l109,l110,l111,l112,l113,l114,l115,l116,l117,
l118,l119,l120,l121,l122,l123,l124,classification_number,index);
##            print('length of l1 = ' ,len(l1))
##            print('length of l2 = ' ,len(l2))
##            print('length of l3 = ' ,len(l3))
##            print('length of l4 = ' ,len(l4))

        except ValueError:
            print(' all files are done reading , just write them in memory now ');
            #save=input('do u want to save new connections[1] or not [0]? \n')
            #if save=='1': # SAVING ALL MATRICES
            
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l1',l1)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l2',l2)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l3',l3)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l4',l4)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l5',l5)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l6',l6)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l7',l7)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l8',l8)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l9',l9)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l10',l10)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l11',l11)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l12',l12)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l13',l13)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l14',l14)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l15',l15)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l16',l16)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l17',l17)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l18',l18)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l19',l19)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l20',l20)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l21',l21)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l22',l22)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l23',l23)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l24',l24)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l25',l25)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l26',l26)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l27',l27)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l28',l28)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l29',l29)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l30',l30)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l31',l31)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l32',l32)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l33',l33)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l34',l34)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l35',l35)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l36',l36)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l37',l37)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l38',l38)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l39',l39)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l40',l40)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l41',l41)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l42',l42)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l43',l43)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l44',l44)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l45',l45)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l46',l46)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l47',l47)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l48',l48)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l49',l49)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l50',l50)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l51',l51)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l52',l52)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l53',l53)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l54',l54)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l55',l55)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l56',l56)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l57',l57)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l58',l58)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l59',l59)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l60',l60)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l61',l61)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l62',l62)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l63',l63)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l64',l64)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l65',l65)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l66',l66)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l67',l67)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l68',l68)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l69',l69)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l70',l70)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l71',l71)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l72',l72)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l73',l73)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l74',l74)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l75',l75)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l76',l76)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l77',l77)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l78',l78)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l79',l79)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l80',l80)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l81',l81)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l82',l82)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l83',l83)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l84',l84)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l85',l85)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l86',l86)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l87',l87)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l88',l88)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l89',l89)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l90',l90)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l91',l91)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l92',l92)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l93',l93)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l94',l94)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l95',l95)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l96',l96)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l97',l97)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l98',l98)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l99',l99)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l100',l100)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l101',l101)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l102',l102)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l103',l103)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l104',l104)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l105',l105)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l106',l106)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l107',l107)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l108',l108)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l109',l109)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l110',l110)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l111',l111)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l112',l112)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l113',l113)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l114',l114)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l115',l115)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l116',l116)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l117',l117)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l118',l118)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l119',l119)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l120',l120)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l121',l121)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l122',l122)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l123',l123)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\l124',l124)


            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\cm1_value',cm1_value)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\cm1_length',cm1_length)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\connection_matrix_2',connection_matrix_2)
            numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\connection_matrix_2_img',connection_matrix_2_img)
            
            print('All connections saved')
    
            break;
        

    # TESTING
    input('STARTING TESTING \n\n')
    fin2=open(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\Mean_Matrix_small_test\FINAL_MAT_small_test.txt','r')
    count=0
    while(True): #IT WILL CONTINUE TILL THE MEAN MAT TEXT FILE ENDS AND GIVES ERROR LOL.
        try:
            (value1,value2,value3,value4,length1,length2,length3,length4,classification_number)=CLASSIFICATION_SUPPORT.KP_Read_1Img_MeanMat(fin2,cm1_value,cm1_length,KP_on_bits_sdr,KP_numOfOnes_in_values,KP_numOfOnes_in_group_length,KP_max_group_size,KP_distinct_values)
            
            sdr_matrix1=numpy.zeros((KP_no_of_connections_per_Neuron,KP_sdr_output_columns)) #SDR_MAT OF A QUADRANT 1
            sdr_matrix2=numpy.zeros((KP_no_of_connections_per_Neuron,KP_sdr_output_columns)) #SDR_MAT OF A QUADRANT 2
            sdr_matrix3=numpy.zeros((KP_no_of_connections_per_Neuron,KP_sdr_output_columns)) #SDR_MAT OF A QUADRANT 3
            sdr_matrix4=numpy.zeros((KP_no_of_connections_per_Neuron,KP_sdr_output_columns)) #SDR_MAT OF A QUADRANT 4
            
            CLASSIFICATION_SUPPORT.KP_Conv_list_to_Matrix(value1,length1,sdr_matrix1,len(value1),len(length1)) #CONVERTING LISTS (VALUE AND LENGHT INTO SDR MATRIX AND ARRANGING VALUE IN FIRST 20 ROWS AND LENGTH IN LAST 20 ROWS FOR EVERY QUAD SDR MAT
            CLASSIFICATION_SUPPORT.KP_Conv_list_to_Matrix(value2,length2,sdr_matrix2,len(value2),len(length2))# sdr_matrix2 IS AUTOMATICALLY CHANGED INSIDE BY THE FUNCTION. ITS PASS BY REFERENCE
            CLASSIFICATION_SUPPORT.KP_Conv_list_to_Matrix(value3,length3,sdr_matrix3,len(value3),len(length3))
            CLASSIFICATION_SUPPORT.KP_Conv_list_to_Matrix(value4,length4,sdr_matrix4,len(value4),len(length4))

            quad1_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2,KP_sdr_output_columns,sdr_matrix1,KP_on_bits_sdr,KP_no_of_connections_per_Neuron,KP_max_noOf_groups_perQuad,1)
            quad2_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2,KP_sdr_output_columns,sdr_matrix2,KP_on_bits_sdr,KP_no_of_connections_per_Neuron,KP_max_noOf_groups_perQuad,1)
            quad3_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2,KP_sdr_output_columns,sdr_matrix3,KP_on_bits_sdr,KP_no_of_connections_per_Neuron,KP_max_noOf_groups_perQuad,1)
            quad4_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2,KP_sdr_output_columns,sdr_matrix4,KP_on_bits_sdr,KP_no_of_connections_per_Neuron,KP_max_noOf_groups_perQuad,1)

            #CONVERTING 4 QUADRANT REPRESENTATIONS INTO 1 IMAGE REPRESENTATION IN ORDER OF QUAD[1 2 3 4]
            sdr_matrix_final=numpy.zeros((KP_numOfQuadrantReprsntatn,KP_sdr_output_columns)) #FINAL IMAGE SDR MATRIX . SINCE WE HAVE 4 QUADRANTS NOW!!
            CLASSIFICATION_SUPPORT.Put_QuadReprntatn_in_SDR(quad1_rep,quad2_rep,quad3_rep,quad4_rep,sdr_matrix_final) #sdr_mat HAS BEEN SET
            IMG_rep=CLASSIFICATION_SUPPORT.KP_Value_Mat(connection_matrix_2_img,KP_sdr_output_columns,sdr_matrix_final,KP_on_bits_sdr,KP_numOfQuadrantReprsntatn,4,0) #HERE FINAL SDR MATRIX WILL HAVE 1 REP PER QUAD. SO TOTAL 4 ROWS OF INPUTS
            if sum(IMG_rep)==0:
                print(sdr_matrix_final)
                print(quad1_rep)
                print(quad2_rep)
                print(quad3_rep)
                print(quad4_rep)
                print(value1,'\n',value2,'\n',value3,'\n',value4,'\n',length1,'\n',length2,'\n',length3,'\n',length4)
                input('is someting wrong with image representation. why is 000000')
            
            print(IMG_rep)
            print('count of images',count)
            count=count+1

            index=CLASSIFICATION_SUPPORT.KP_Extract_Indexes_ofOnes(KP_on_bits_sdr,KP_sdr_output_columns,IMG_rep) # GIVES INDEXES OF ALL 1'S
            
            
            l_max_true=CLASSIFICATION_SUPPORT.Predicting_Class(l1,l2,l3,l4,index)
            print('l_max_true = ',l_max_true)

        except ValueError:
            print(' All test images are over.. done!! ');
            break;
    
        
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
