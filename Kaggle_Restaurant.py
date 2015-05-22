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

##    noOfInputsInCityName=34 #TOTAL 34 CITIES IN TRAIN. ALL NEW CITIES IN TEST SHOULD BE 0 IF NOT FOUND IN TRAIN. THEY SHALL BE 0 SO THAT THEY DO NOT DISTURB THE FINAL REPRESENTATION
##    noOfInputsInMonthsYears=25   # 217/10 = 22 DISTINCT INPUTS. BUT WE TAKE 25 ROUNDLY
##    noOfInputsInCityGroup=2
##    noOfInputsInType=3 # SINCE THERE ARE NO MB TYPES IN TRAIN BUT THERE ARE 291 MB TYPES IN TEST.WE ONLY CONSIDER TRAIN DATA AND NEW DATA IN TEST IS 0 
##    noOfInputsInP1_P37=25
    
    percentage_of_on_bits_sdr=2
    no_of_connections_per_Neuron=42 #VERY IMPORTANT. #39 +3. IT IS ONLY USED TO SET THE CM2 WHICH IS A bunch OF INDEXES OF CONNECTIONS FOR ALL NEURONS.EVEN IF ITS > THEN THE ACTUAL INPUT COUNT ,MY EXCELLENT PROGRAMING SKILLS TAKE CARE OF IT. MY HEALTH VALUE MAT PROGRESSES INPUTS BASED ON ACTUAL INPUT SDR LENGTH. AND GIVES OUTPUT WHEN ITS RIGHT
                                    #SO JUST SET no_of_connections_per_Neuron ANYTHING ABOVE ACTUAL NUMBER OF INPUTS.
    input_weightage=numpy.ones((no_of_connections_per_Neuron)) #IF ANY PARTICULAR INPUT IS IMPORTANT WE INCREASE ITS WEIGHTAGE. SO CHANGE IN THAT INPUT DEFINETLY AFFECTS THE OUTPUT. WE SET THEN INITIALLY TO 1 SO ORIGINAL SDR VALUE REMAINS INTACT.FURTHER WE CHANGE SOME VALUES IN THE WEIGHTAGE FUNCTION BELOW
    on_bits_sdr=math.floor((percentage_of_on_bits_sdr) * (sdr_output_columns/100))
    total_noOf_classes=139 #TOTAL CLASSES FOR CLASSIFICATION
    maxNoOfDistinctExclusivPatternsPossible=sdr_output_columns/on_bits_sdr #FOR ANY GIVEN SDR LENGTH AND % OF ON BITS
    
##    cm1_CityName=numpy.random.rand(sdr_output_columns,(numOfOnesInBin*noOfInputsInCityName)) # USED TO CONVERT BINARY TO SDR. 
##    cm1_MonthYear=numpy.random.rand(sdr_output_columns,(numOfOnesInMonthsYears+noOfInputsInMonthsYears)) # MONTHyEAR BITS OVERLAP SO TOTAL BITS IS NO_OFoNES+TOTALiNPUTS
##    cm1_CityGroup=numpy.random.rand(sdr_output_columns,(numOfOnesInBin*noOfInputsInCityGroup)) # USED TO CONVERT BINARY TO SDR. 2 MUTUALLY EXCLUSIVE INPUTS. SO 10 BIT REPRESNTATION
##    cm1_Type=numpy.random.rand(sdr_output_columns,(numOfOnesInBin*noOfInputsInType)) # 3 MUTUALLY EXCLUSIVE INPUTS. SO 15 BIT REPRESNTATION
##    cm1_p1_p37=numpy.random.rand(sdr_output_columns,(numOfOnesInP1_P37+noOfInputsInP1_P37)) # 25 TOTAL DISTINCT INPUTS IN P1-P37
    cm2=CLASSIFICATION_SUPPORT.KP_Connection_Mat_2(sdr_output_columns,no_of_connections_per_Neuron) # ITS NOT A REGULAR cm. sdr_output_columns HAS TO BE > no_of_connections_per_Neuron
    x=[]
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
    l125=[]
    l126=[]
    l127=[]
    l128=[]
    l129=[]
    l130=[]
    l131=[]
    l132=[]
    l133=[]
    l134=[]
    l135=[]
    l136=[]
    l137=[]
    l138=[]
    l139=[]

    list_of_list=[]

    ###########  TRAIN
    f=open(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\train2.csv','rt') #C:\Users\ahegshetye\Downloads\clt\kag\restaurnt\train.csv
    reader=csv.reader(f)
    for row in reader:
        x.append(row)


    for excel_rows in range(1,len(x)): #0TH INDEX HAS TITLES
        ip=[]
        sdr_rep=[]

        #OPEN DATE MONTH_YEAR
        ip.append(math.ceil((int(x[excel_rows][2])/10)))# DIVIDE BY 10 SO THAT FOR 10 MONTHS IT HAS SAME VALUE. 

        #OPEN DATE YEAR
        #ip.append(x[excel_rows][3])

        #CITY NAME
        ip.append(int(x[excel_rows][4]))


        #CITY GROUP
        if x[excel_rows][5]=='Big Cities':
            ip.append(1)
        elif x[excel_rows][5]=='Other':
            ip.append(2)
        else:
            print(excel_rows)
            print(x[excel_rows][5])
            input(' 3rd unexpected city group found. ExCEL IS OVER. LETS BREAK')
            break


        #TYPE -  FC: Food Court=1, IL: Inline=2, DT: Drive Thru=3, MB: Mobile=4
        if x[excel_rows][6]=='FC':
            ip.append(1)
        elif x[excel_rows][6]=='IL':
            ip.append(2)
        elif x[excel_rows][6]=='DT':
            ip.append(3)
        elif x[excel_rows][6]=='MB':
            ip.append(0)
        else:
            input('unexpected TYPE ')


        #PA1 - PA37
        for i in range(7,44):
            ip.append(math.ceil(float(x[excel_rows][i])))
            



        #REVENUE
        ip.append(int(x[excel_rows][45])) #45TH COLUMN HAS THE REVENUE CLASS
        #ip.append(math.floor(int(x[excel_rows][45])/1000000) ) #45th COLUMN HAS THE REVENUE AMOUNT

        print(ip)
        input(' IP LOOKS GOOD? ')

##### SDR'S

        # FOR 8 ON BIT ONE SDR 400 BITS TOTAL. 50 TOTAL EXCLUSIVE REPRESENTATIONS POSSIBLE. cITY GROUP NEVER WILL BE 0.

        
        #MONTH_YEAR
        sdr_rep.append(CLASSIFICATION_SUPPORT.KR_Converting_input_2SDR(sdr_output_columns,4,ip[0],1))
        
##        bin_rep1=(CLASSIFICATION_SUPPORT.Health_Converting_input_2binary(noOfInputsInMonthsYears,numOfOnesInMonthsYears,ip[0],1)) 
##        sdr_rep.append(CLASSIFICATION_SUPPORT.Binary2Sdr(bin_rep1,cm1_MonthYear,on_bits_sdr))

        #CITY NAME
        
        sdr_rep.append(CLASSIFICATION_SUPPORT.KR_Converting_input_2SDR(sdr_output_columns,on_bits_sdr,ip[1],0))
     
        
##        bin_rep1=(CLASSIFICATION_SUPPORT.Health_Converting_input_2binary(noOfInputsInCityName,numOfOnesInBin,ip[1],0)) 
##        sdr_rep.append(CLASSIFICATION_SUPPORT.Binary2Sdr(bin_rep1,cm1_CityName,on_bits_sdr))

##      #CITY GROUP

      
        #CITY GROUP IS NEVER 0 SO ADDED IP[] WITH SOMETHING IS FINE DURING SDR CONVERSION
        sdr_rep.append(CLASSIFICATION_SUPPORT.KR_Converting_input_2SDR(sdr_output_columns,on_bits_sdr,(maxNoOfDistinctExclusivPatternsPossible-10+ip[2]),0)) #40 + INPUT  SO IT MAY BE 41 OR 42
        
##        bin_rep1=(CLASSIFICATION_SUPPORT.Health_Converting_input_2binary(noOfInputsInCityGroup,numOfOnesInBin,ip[2],0)) 
##        sdr_rep.append(CLASSIFICATION_SUPPORT.Binary2Sdr(bin_rep1,cm1_CityGroup,on_bits_sdr))


##      #TYPE
        #WE ARE MULTIPLYING IP[] SO THAT IF ITS 0 THEN OUTPUT WILL REMAIN 0.. NO EXTRA NOISE WILL BE ADDED.
        sdr_rep.append(CLASSIFICATION_SUPPORT.KR_Converting_input_2SDR(sdr_output_columns,on_bits_sdr,(math.floor((maxNoOfDistinctExclusivPatternsPossible/3)*ip[3])),0)) 
        
##        bin_rep2=(CLASSIFICATION_SUPPORT.Health_Converting_input_2binary(noOfInputsInType,numOfOnesInBin,ip[3],0)) 
##        sdr_rep.append(CLASSIFICATION_SUPPORT.Binary2Sdr(bin_rep2,cm1_Type,on_bits_sdr))


##      #P1 - P37
        
        #25 IS MAX VALUE IN P'S
        for i in range(4,41): #ip[41] HAS THE CLASS VALUE

            sdr_rep.append(CLASSIFICATION_SUPPORT.KR_Converting_input_2SDR(sdr_output_columns,4,((math.ceil(i/3))*ip[i]),1))
##            
##            bin_rep3=(CLASSIFICATION_SUPPORT.Health_Converting_input_2binary(noOfInputsInP1_P37,numOfOnesInP1_P37,math.floor(i/3)+ip[i],1))  #OVERLAPPING BITS
##            sdr_rep.append(CLASSIFICATION_SUPPORT.Binary2Sdr(bin_rep3,cm1_p1_p37,on_bits_sdr))
##        

        CLASSIFICATION_SUPPORT.KR_WeightedSDR(sdr_rep,input_weightage) #MANNUALLY SET input_weightage INSIDE THIS FUNCTION. [sdr_rep = sdr_rep * input_weightage]
        
        final_rep=CLASSIFICATION_SUPPORT.Health_Value_Mat(cm2,sdr_output_columns,on_bits_sdr,sdr_rep)
        #input(final_rep)
        


        index=CLASSIFICATION_SUPPORT.KP_Extract_Indexes_ofOnes(on_bits_sdr,sdr_output_columns,final_rep) # GIVES INDEXES OF ALL 1'S
        CLASSIFICATION_SUPPORT.KR_Classifying(l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18,l19,l20,l21,l22,l23,l24,l25,l26,l27,l28,l29,l30,l31,l32,l33,l34,l35,l36,l37,l38,l39,l40,l41,l42,l43,l44,l45,l46,l47,l48,l49,l50,l51,l52,l53,l54,l55,l56,l57,l58,l59,l60,l61,l62,l63,l64,l65,l66,l67,l68,l69,l70,l71,l72,l73,l74,l75,l76,l77,l78,l79,l80,l81,l82,l83,l84,l85,l86,l87,l88,l89,l90,l91,l92,l93,l94,l95,l96,l97,l98,l99,l100,l101,l102,l103,l104,l105,l106,l107,l108,l109,l110,l111,l112,l113,l114,l115,l116,l117,
l118,l119,l120,l121,l122,l123,l124,l125,l126,l127,l128,l129,l130,l131,l132,l133,l134,l135,l136,l137,l138,l139,ip[41],index)    #ip[41] HAS THE CLASS VALUE
         
        
    CLASSIFICATION_SUPPORT.KR_Converting_Lists2_ListOfList(l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18,l19,l20,l21,l22,l23,l24,l25,l26,l27,l28,l29,l30,l31,l32,l33,l34,l35,l36,l37,l38,l39,l40,l41,l42,l43,l44,l45,l46,l47,l48,l49,l50,l51,l52,l53,l54,l55,l56,l57,l58,l59,l60,l61,l62,l63,l64,l65,l66,l67,l68,l69,l70,l71,l72,l73,l74,l75,l76,l77,l78,l79,l80,l81,l82,l83,l84,l85,l86,l87,l88,l89,l90,l91,l92,l93,l94,l95,l96,l97,l98,l99,l100,l101,l102,l103,l104,l105,l106,l107,l108,l109,l110,l111,l112,l113,l114,l115,l116,l117,
l118,l119,l120,l121,l122,l123,l124,l125,l126,l127,l128,l129,l130,l131,l132,l133,l134,l135,l136,l137,l138,l139,list_of_list)
    input('DO U WANT TO SAVE THIS LEARNING AND OVERWRITE PREVIOUS LEARNT')
##    numpy.save(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\cm1_CityGroup',cm1_CityGroup)
##   numpy.save(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\cm1_Type',cm1_Type)
##    numpy.save(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\cm1_p1_p37',cm1_p1_p37)
    numpy.save(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\cm2',cm2)
    numpy.save(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\list_of_list',list_of_list)
    
##    numpy.save(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\cm1_CityName',cm1_CityName)
##    numpy.save(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\cm1_MonthYear',cm1_MonthYear)

    
######### #TESTING  USE SEPARATE FUNCTION FOR TESTING.. THAT KR TESTING WORKS FINE.. USE THAT. MAINTAINING TWO FUNCTIONS IS TIME CONSUMING.

##    
##    x2=[]
##    f2=open(r'C:\Users\ahegshetye\Downloads\clt\kag\restaurnt\test2.csv','rt') #C:\Users\ahegshetye\Downloads\clt\kag\restaurnt\train.csv
##    reader=csv.reader(f2)
##    for row in reader:
##        x2.append(row)
##
##    
##    for excel_rows in range(1,len(x2)): #0TH INDEX HAS TITLES
##        ip=[]
##        sdr_rep=[]
##
##        #OPEN DATE MONTH_YEAR
##        ip.append(math.ceil((int(x2[excel_rows][2])/10)))# DIVIDE BY 5 SO THAT FOR 5 MONTHS IT HAS SAME VALUE. 
##
##        #OPEN DATE YEAR
##        #ip.append(x2[excel_rows][3])
##
##        #CITY NAME
##        ip.append(int(x2[excel_rows][4]))
##
##
##        #CITY GROUP
##        if x2[excel_rows][5]=='Big Cities':
##            ip.append(1)
##        elif x2[excel_rows][5]=='Other':
##            ip.append(2)
##        else:
##            print(excel_rows)
##            print(x2[excel_rows][5])
##            input(' 3rd unexpected city group found. ExCEL IS OVER. LETS BREAK')
##            break
##
##
##        #TYPE -  FC: Food Court=1, IL: Inline=2, DT: Drive Thru=3, MB: Mobile=4
##        if x2[excel_rows][6]=='FC':
##            ip.append(1)
##        elif x2[excel_rows][6]=='IL':
##            ip.append(2)
##        elif x2[excel_rows][6]=='DT':
##            ip.append(3)
##        elif x2[excel_rows][6]=='MB':
##            ip.append(0)
##        else:
##            input('unexpected TYPE ')
##
##
##        #PA1 - PA37
##        for i in range(7,44):
##            ip.append(math.ceil(float(x2[excel_rows][i])))
##            
##
##
##
##        #REVENUE
##        ip.append(int(x2[excel_rows][45])) #45TH COLUMN HAS THE REVENUE CLASS
##        #ip.append(math.floor(int(x[excel_rows][45])/1000000) ) #45th COLUMN HAS THE REVENUE AMOUNT
##
##        print(ip)
##        input(' IP LOOKS GOOD? ')
##        print(cm2)
##        print(' cm2 looks good? ')
##        print(list_of_list)
##        print(' list_of_list looks good? ')
##        
##
###### SDRS
##        
##
##        # FOR 8 ON BIT ONE SDR 400 BITS TOTAL. 50 TOTAL EXCLUSIVE REPRESENTATIONS POSSIBLE. cITY GROUP NEVER WILL BE 0.
##
##        
##        #MONTH_YEAR
##        sdr_rep.append(CLASSIFICATION_SUPPORT.KR_Converting_input_2SDR(sdr_output_columns,on_bits_sdr,ip[0],1))
##        
##
##        #CITY NAME
##        
##        sdr_rep.append(CLASSIFICATION_SUPPORT.KR_Converting_input_2SDR(sdr_output_columns,on_bits_sdr,ip[1],0))
##     
##        
####      #CITY GROUP
##
##      
##        #CITY GROUP IS NEVER 0 SO ADDED IP[] WITH SOMETHING IS FINE DURING SDR CONVERSION
##        sdr_rep.append(CLASSIFICATION_SUPPORT.KR_Converting_input_2SDR(sdr_output_columns,on_bits_sdr,(maxNoOfDistinctExclusivPatternsPossible-10+ip[2]),0)) #40 + INPUT  SO IT MAY BE 41 OR 42
##        
##
####      #TYPE
##        #WE ARE MULTIPLYING IP[] SO THAT IF ITS 0 THEN OUTPUT WILL REMAIN 0.. NO EXTRA NOISE WILL BE ADDED.
##        sdr_rep.append(CLASSIFICATION_SUPPORT.KR_Converting_input_2SDR(sdr_output_columns,on_bits_sdr,(math.floor((maxNoOfDistinctExclusivPatternsPossible/3)*ip[3])),0)) 
##        
##
####      #P1 - P37
##        
##        #25 IS MAX VALUE IN P'S
##        for i in range(4,41): #ip[41] HAS THE CLASS VALUE
##
##            sdr_rep.append(CLASSIFICATION_SUPPORT.KR_Converting_input_2SDR(sdr_output_columns,on_bits_sdr,((math.ceil(i/3))*ip[i]),1))
####      
##
##        CLASSIFICATION_SUPPORT.KR_WeightedSDR(sdr_rep,input_weightage) #MANNUALLY SET input_weightage INSIDE THIS FUNCTION. [sdr_rep = sdr_rep * input_weightage]
##        
##        final_rep=CLASSIFICATION_SUPPORT.Health_Value_Mat(cm2,sdr_output_columns,on_bits_sdr,sdr_rep)
##        #input(final_rep)
##        
##
##
##        index=CLASSIFICATION_SUPPORT.KP_Extract_Indexes_ofOnes(on_bits_sdr,sdr_output_columns,final_rep) # GIVES INDEXES OF ALL 1'S
##       
##
##        l_max_true=CLASSIFICATION_SUPPORT.KR_Predicting_Class(list_of_list,index,on_bits_sdr,total_noOf_classes)
##        print(l_max_true)
##        print(l_max_true.argmax()+1)
##



    
if __name__ == '__main__':
  main()
