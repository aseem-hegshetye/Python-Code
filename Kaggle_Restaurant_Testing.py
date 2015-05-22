#CLASSIFYING CAR CONDITION
import sys
import operator
import math
import time
import numpy
import csv
#from numpy import *

from TALK_support import *
from TALK_support2 import *
from CLASSIFICATION_SUPPORT import *
def main():

    sdr_output_columns=400
    
##    numOfOnesInMonthsYears=5 #  BIT OVERLAPPING. SO CONSECUTIVE MONTHS WILL HAVE LITTLE BIT SAME REPRESENTATION
##    numOfOnesInBin=2
##    numOfOnesInP1_P37=5
##    numOfOnesInCityNames=2
##    numOfOnesInCityGroup=2
##
##    noOfInputsInCityName=34 #TOTAL 34 CITIES IN TRAIN. ALL NEW CITIES IN TEST SHOULD BE 0 IF NOT FOUND IN TRAIN. THEY SHALL BE 0 SO THAT THEY DO NOT DISTURB THE FINAL REPRESENTATION
##    noOfInputsInMonthsYears=25   
##    noOfInputsInCityGroup=2
##    noOfInputsInType=3 # SINCE THERE ARE NO MB TYPES IN TRAIN BUT THERE ARE 291 MB TYPES IN TEST.WE ONLY CONSIDER TRAIN DATA AND NEW DATA IN TEST IS 0 
##    noOfInputsInP1_P37=25
    
    percentage_of_on_bits_sdr=2
    no_of_connections_per_Neuron=42 #VERY IMPORTANT. #39 +3
    input_weightage=numpy.ones((no_of_connections_per_Neuron)) #IF ANY PARTICULAR INPUT IS IMPORTANT WE INCREASE ITS WEIGHTAGE. SO CHANGE IN THAT INPUT DEFINETLY AFFECTS THE OUTPUT. WE SET THEN INITIALLY TO 1 SO ORIGINAL SDR VALUE REMAINS INTACT.FURTHER WE CHANGE SOME VALUES IN THE WEIGHTAGE FUNCTION BELOW
    on_bits_sdr=math.floor((percentage_of_on_bits_sdr) * (sdr_output_columns/100))
    total_noOf_classes=139 #TOTAL CLASSES FOR CLASSIFICATION
    maxNoOfDistinctExclusivPatternsPossible=sdr_output_columns/on_bits_sdr #FOR ANY GIVEN SDR LENGTH AND % OF ON BITS
    
##    cm1_CityGroup=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\cm1_CityGroup.npy')
##    cm1_Type=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\cm1_Type.npy')
##    cm1_p1_p37=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\cm1_p1_p37.npy')
    cm2=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\cm2.npy')
    list_of_list=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\list_of_list.npy')
##    cm1_CityName=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\cm1_CityName.npy')
##    cm1_MonthYear=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\cm1_MonthYear.npy')
    x2=[]
    output=[]
    count=0
#TESTING
    
    
    f=open(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\test2.csv','rt') #C:\Users\ahegshetye\Downloads\clt\kag\restaurnt\train.csv
    reader=csv.reader(f)
    for row in reader:
        x2.append(row)

    
    for excel_rows in range(1,len(x2)): #0TH INDEX HAS TITLES
        ip=[]
        sdr_rep=[]

        #OPEN DATE MONTH_YEAR
        ip.append(math.ceil((int(x2[excel_rows][2])/10)))# DIVIDE BY 5 SO THAT FOR 5 MONTHS IT HAS SAME VALUE. 

        #OPEN DATE YEAR
        #ip.append(x2[excel_rows][3])

        #CITY NAME
        if x2[excel_rows][4] =='#N/A': #IF CITY DOESNT EXIST PUT 0
            ip.append(0)
        else:
            ip.append(int(x2[excel_rows][4]))


        #CITY GROUP
        if x2[excel_rows][5]=='Big Cities':
            ip.append(1)
        elif x2[excel_rows][5]=='Other':
            ip.append(2)
        else:
            print(excel_rows)
            print(x2[excel_rows][5])
            input(' 3rd unexpected city group found. ExCEL IS OVER. LETS BREAK')
            break


        #TYPE -  FC: Food Court=1, IL: Inline=2, DT: Drive Thru=3, MB: Mobile=4
        if x2[excel_rows][6]=='FC':
            ip.append(1)
        elif x2[excel_rows][6]=='IL':
            ip.append(2)
        elif x2[excel_rows][6]=='DT':
            ip.append(3)
        elif x2[excel_rows][6]=='MB': # THIS TYPE IS NOT THERE IN TRAIN SO LETS NOT DISTURB THE REPRESENTATION
            ip.append(0)
        else:
            input('unexpected TYPE ')


        #PA1 - PA37
        for i in range(7,44):
            ip.append(math.ceil(float(x2[excel_rows][i])))
            



        
##        print(ip)
##        input(' IP LOOKS GOOD? ')
##        print(cm2)
##        print(' cm2 looks good? ')
##        print(list_of_list)
##        print(' list_of_list looks good? ')

      
#### SDRS
        

        # FOR 8 ON BIT ONE SDR 400 BITS TOTAL. 50 TOTAL EXCLUSIVE REPRESENTATIONS POSSIBLE. cITY GROUP NEVER WILL BE 0.

        
        #MONTH_YEAR
        sdr_rep.append(CLASSIFICATION_SUPPORT.KR_Converting_input_2SDR(sdr_output_columns,4,ip[0],1))
        

        #CITY NAME
        
        sdr_rep.append(CLASSIFICATION_SUPPORT.KR_Converting_input_2SDR(sdr_output_columns,on_bits_sdr,ip[1],0))
     
        
##      #CITY GROUP

      
        #CITY GROUP IS NEVER 0 SO ADDED IP[] WITH SOMETHING IS FINE DURING SDR CONVERSION
        sdr_rep.append(CLASSIFICATION_SUPPORT.KR_Converting_input_2SDR(sdr_output_columns,on_bits_sdr,(maxNoOfDistinctExclusivPatternsPossible-10+ip[2]),0)) #40 + INPUT  SO IT MAY BE 41 OR 42
        

##      #TYPE
        #WE ARE MULTIPLYING IP[] SO THAT IF ITS 0 THEN OUTPUT WILL REMAIN 0.. NO EXTRA NOISE WILL BE ADDED.
        sdr_rep.append(CLASSIFICATION_SUPPORT.KR_Converting_input_2SDR(sdr_output_columns,on_bits_sdr,(math.floor((maxNoOfDistinctExclusivPatternsPossible/3)*ip[3])),0)) 
        

##      #P1 - P37
        
        #25 IS MAX VALUE IN P'S
        for i in range(4,41): #ip[41] HAS THE CLASS VALUE

            sdr_rep.append(CLASSIFICATION_SUPPORT.KR_Converting_input_2SDR(sdr_output_columns,4,((math.ceil(i/3))*ip[i]),1))
##   

        CLASSIFICATION_SUPPORT.KR_WeightedSDR(sdr_rep,input_weightage) #MANNUALLY SET input_weightage INSIDE THIS FUNCTION. [sdr_rep = sdr_rep * input_weightage]
        
        final_rep=CLASSIFICATION_SUPPORT.Health_Value_Mat(cm2,sdr_output_columns,on_bits_sdr,sdr_rep)
        input(final_rep)
        


        index=CLASSIFICATION_SUPPORT.KP_Extract_Indexes_ofOnes(on_bits_sdr,sdr_output_columns,final_rep) # GIVES INDEXES OF ALL 1'S
       

        
      

        l_max_true=CLASSIFICATION_SUPPORT.KR_Predicting_Class(list_of_list,index,on_bits_sdr,total_noOf_classes)
        #print(l_max_true)
        print(l_max_true.argmax()+1)
        #output.append((l_max_true.argmax()+1))
        #count=count+1
        #print(count)
        
    #numpy.save(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\Revenue_Prediction',output) # !!!!!!!!!!UNCOMENT
    #print('output HAS BEEN SAVED !! ')
    
    #print(output)
    #print((l_max_true.argmax()+1)*1000000)
    #output=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\Revenue_Prediction.npy') # !!!!!!!!



    
if __name__ == '__main__':
  main()
