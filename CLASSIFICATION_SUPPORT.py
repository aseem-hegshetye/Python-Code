import sys
import operator
import math
import time
import numpy
import random
import csv
#from numpy import *

from TALK_support import *
from TALK_support2 import *

class CLASSIFICATION_SUPPORT():
    ascii_matrix_columns=10  # DONT CHANGE. its the length of binary input. i have written a code in SENTENCE () class to set it up.its not flexible
                                #HERE [SENTENCE.ascii_matrix_columns] IS USED FOR NO. OF COLUMNS IN ASCII.
    sdr_output_columns=80 # number of columns in the output
    percentage_of_on_bits_sdr=10
    on_bits_sdr=math.floor((percentage_of_on_bits_sdr) * (sdr_output_columns/100))
    no_of_connections_per_Neuron=5 #MAXIMUN LENGTH OF SEQUENCE THAT CAN BE CLASSIFIED CORRECTLY. BECAUSE ONLY AFTER THESE MANY COUNTS WILL THE RESPONSE GO IN OUTPUT
                                    #IF INPUT IS 4 SEQ LONG, [THEN no_of_connections_per_Neuron-4] IS ALL WASTED.IT STAYS IN VALUE MATRIX UNTOUCHED.
    filename='car_data.txt' # TEXT FILE THAT HAS DATA
    no_of_lines_toRead=800 # NUMBER OF LINES TO READ FROM THE TEXT FILE


    

    #KP = KAGGLE_PLANKATON
##    KP_distinct_values=4;
##    KP_max_group_size=30; # SAME VARIABLE IN MATLAB. ACTUALLY max_group_size IS 2 TO 30 SO 28. BUT WE WILL KEEP 30. IT IS HOW MANY PIXELS CAN BE AT MAX IN A GROUP
##    KP_numOfOnes_in_values=2;
##    KP_numOfOnes_in_group_length=4; #GIVES GOOD OVERLAP 12 AND 9 WILL HAVE 1 SIMILAR BIT.
##    KP_sdr_output_columns=100
##    KP_percentage_of_on_bits_sdr=10
##    KP_on_bits_sdr=math.floor((KP_percentage_of_on_bits_sdr) * (KP_sdr_output_columns/100))
##    KP_no_of_connections_per_Neuron=40 #MAXIMUN LENGTH OF SEQUENCE THAT CAN BE CLASSIFIED CORRECTLY. BECAUSE ONLY AFTER THESE MANY COUNTS WILL THE RESPONSE GO IN OUTPUT
                                    #IF INPUT IS 4 SEQ LONG, [THEN no_of_connections_per_Neuron-4] IS ALL WASTED.IT STAYS IN VALUE MATRIX UNTOUCHED. max_no_ofGroups=50 PER IMAGE IS IN MATLAB.
                                    #KP_no_of_connections_per_Neuron IS PER QUADRANT. BUT THERE ARE TWO PARAMETERS PER INPUT GROUP.VALUE(/\-|) AND LENGTH.

######----------------------------------------------------------------------------------------------------------------------------
    #KAGGLE RESTAURANT KR

        # CLASSIFICATION_SUPPORT.KR_Extract_Indexes_ofOnes
    def KR_Extract_Indexes_ofOnes (on_bits_sdr,sdr_output_columns,sdr,test_train): #GIVES INDEXES OF ALL ONES FOR ONE PARTICULAR sdr VECTOR
        #TESTED AND WORKING !!
        #test_train =1 = TEST ... test_train =2 = TRAIN
        indx=numpy.zeros((on_bits_sdr))
        count=0

        #TRAIN
        if test_train ==2 : #IN TRAIN THERE CANNOT BE ANY 0 INPUT. IF THERE IS ONE WE NEED TO FIX IT 
            if sum(sdr) != on_bits_sdr: # IN TRAIN - IF THERE ARE 8 ON_BITS_SDR THEN  SUM OF ALL SDR BITS SHOULD BE 8. ELSE THERE IS A PROBLEM
                input(' PROBLEMBO IN TRAIN SDR MOTHER FUCKER. ALL SDR BITS DONT ADD TO ON_BITS_SDR VALUE ')
            else: #PICK ALL ONES FROM SDR
                for i in range (0,sdr_output_columns):
                    if sdr[i]==1:
                        indx[count]=i
                        count=count+1

        #TEST
        if test_train ==1 : #IN TEST IF SUM(SDR)==0 THEN WE SET THE INDEXES TO -1 SO THAT THEY DONT OVERLAP WITH TRAINED LERANT NEURONS CONNECTION PATTERN
            if sum(sdr)==0:
                indx=indx-1 #SETTING ALL INDEXS TO -1
            elif sum(sdr)==on_bits_sdr: # PICK ALL INDEXES OF ONES FROM THE SDR
                for i in range (0,sdr_output_columns):
                    if sdr[i]==1:
                        indx[count]=i
                        count=count+1
            else:
                input( ' YOU NEED TO HAVE on_bits_sdr NUMBER OF 1S IN THE SDR OR ALL BITS CAN BE ZERO. \n BUT HERE FEW BITS ARE ON AND ARE NOT EQUAL TO on_bits_sdr IN " TEST " ')
                
            
        
        
            
        return(list(indx)) #CONVERTING INDEX FROM ARRAY TO LIST FOR FURTHER EASE OF PROCESSING
    
    
    def KR_Converting_input_2SDR(n,m,k,exc_overlap): # CONVERTING IP TO SDR DIRECTLY IN ORDER TO AVOID THE DISCREPENCIES CREATED BY COMPUTERS RANDOM SDR CONNECTION MATRIX. OVER LAPPING BITS IN BINARY DONT RETAIN SAME SHARED BEHAVIOUR WHEN CONVERTED TO SDR. IF THERE ARE 5 BITS IN BIN OVERLAPPING THEN SDR WONT NECESARRY HAVE 5 OVERLAPPING BITS. THATS BAD FOR FUTURE SEMANTIC SIMALIRITIES.
        #I HAVE ADDED EXTRA ZERO CONDITION. IF INPUT IS 0 THEN  RETURN ALL ZEROS
        # n- TOTAL NUMBER OF BITS IN SDR OUTPUT
        # m-NUMBER OF 1'S IN EVERY SDR REPRESENTATION NEEDED
        # k- WHICH INPUT (1ST,2ND,3RD..) IF 0 THE OUTPUT 0
        #exc_overlap- IF 0 THEN MUTUALLY EXCLUSIVE OUTPUT NEEDED. IF 1 THEN OVERLAPING OUTPUT NEEDED
        #IF REPRNTATION > SDR SIZE   OR  INPUT < 0    WE JUST GIVE OUTPUT=0.WE ADJUST THE PARAMETERS ACCORDING TO TRAIN AND ALL NEW UNSEEN INPUTS FROM TEST SHOULD BE MADE 0 SO AS TO AVOID THEM FROM DISTURBING THE FINAL REP
        
        output=numpy.zeros((n))

            
        if k>0: #IF K (NPUT) = 0 or <0 NEGATIVE THEN WE JUST RETURN 0. NO BIN REPRENTATION NEEDED.

            if exc_overlap ==0: #MUTUALY EXCLUSIVE
                if (m*(k-1))+m > n: #IF OUT OF BOUNDS THEN OUTPUT=0
                    input('KR_Converting_input_2SDR INPUT IS GREATER THAN SDR DIMENSION #MUTUALY EXCLUSIVE \n COME FIX IT')
                else:    
                    output[(m*(k-1)):(m*(k-1))+m]=1 # U BETTER SET THIS TO 1 COZ INDEXES ON ONES IS ONLY LOOKING FOR ONES IN THE SDR LOL
            elif exc_overlap ==1:#OVERLAPPING
                if (k-1)+m > n : #IF OUT OF BOUNDS THEN OUTPUT=0
                    input('KR_Converting_input_2SDR INPUT IS GREATER THAN SDR DIMENSION #OVERLAPPING \n COME FIX IT')
                else:
                    output[(k-1) : (k-1)+m]=1

        return output

    

    def KR_Predicting_Class(list_of_list,index,KP_on_bits_sdr,total_classes):# PREDICTING WHAT CLASS THE INPUT BELONGS TO MORE APPROPRIATLY
        #SINCE BUILDING LIST_OF_LIST IS DIFFICULT TO TRY OUT U CAN USE KP_Predicting_Class_trial. JUST GIVE TWO NORMAL LISTS AND IT WILL GIVE U MATCHING COUNT
        #list_of_list and index WILL HAVE INDEXES OF ONES. ALL THOSE INDEXES WILL ALWAYS BE DISTINCT, SO WHILE TESTING MAKE SURE TO USE DISTINCT INDEXES. ORDER DOESNT MATTER IN THIS AWESOME NEW FUNCTION. IT WILL STILL GIVE RIGHT MATCHES
        #eg list_of_list=[1,5,20,40]. THIS IS A PATTEREND LEARNED. index=[1,5,0,22] WILL HAVE 2 MATCHING INDICES OF ONES.index=[1,15,5,22] ALSO HAVE 2 MATCHIN INDICES BUT THE ORDER IS DIFFERENT AS OF THE LEARNT INPUT, BUT STILL THIS FUCNTION WILL GIVE OUTPUT = 2 YAY !!
        l_max_true=numpy.zeros((total_classes)) # NUMBER OF CLASSIFICATIONS = 4.MAXIMUM TRUES IN EACH CLASS. CLASS 1= INDEX 0
                                        #CLASS 2= INDEX 1.. SO ON
        

        for lists in range(0,len(list_of_list)): #FOR L1 L2 L3 ETX..
            for i in range (0,len(list_of_list[lists])):  #FOR L1[0] L1[1] L1[2]..
                count=0
                for list_stored in range (0,KP_on_bits_sdr): # EVERY BIT OF NEW PREDICTING  IMAGE SDR
                    for list_new in range (0,KP_on_bits_sdr): # EVERY BIT OF STORED IMAGE SDR
                        if index[list_stored]== list_of_list[lists][i][list_new] : # CHECKS IF EVERY BIT OF NEW IMAGE IS FOUND IN STORED IMAGE. IF YES COUNT ++
                            count=count+1
                            break
                        #MOSTLY ALL INDEXES ARE IN INCREASING ORDER. SO TO MAKE IT FAST MAY BE IF INDEX VALUE INCREASES JUST BREAK COZ ITS NOT GOING TO MATCH LATER TOO LOL

                if l_max_true[lists]<count: # IF THIS COUNT OF TRUES IS MORE THEN PREVIOUS ONE.. STORE IT 
                    l_max_true[lists]=count
                            
                        

        return(l_max_true)


         #  CLASSIFICATION_SUPPORT.KR_WeightedSDR
    def KR_WeightedSDR(sdr,no_of_connections_per_Neuron): #SETS WEIGHT OF INPUTS AND THEN MULTIPLIES SDR AND GIVES APPROPRIATE REPRESENTATION. HIGH WEIGHT INPUT PRODUCES HIGH VALUE SDR SO CHANGE IN THAT INPUT NOTICIEBLY AFFECTS THE OUTPUT
        #THIS FUNCTION WONT WORK FOR KAGGLE RESTAURANT FUNCTION COZ I HAVE DECLARED WEIGHTS INSIDE THIS FUNCTION NOW. IT WILL WORK FOR TEST_CLASS RESTAURANT
        #no_of_connections_per_Neuron= len(cm2[0]) -- THAT IS THE NUMBER OF PARAMETERS PER CASE
        #DECLARING WEIGHTS INSIDE THIS FUNCTION ITSELF.
        
        
        weight=numpy.ones((no_of_connections_per_Neuron)) #IF ANY PARTICULAR INPUT IS IMPORTANT WE INCREASE ITS WEIGHTAGE. SO CHANGE IN THAT INPUT DEFINETLY AFFECTS THE OUTPUT. WE SET THEN INITIALLY TO 1 SO ORIGINAL SDR VALUE REMAINS INTACT.FURTHER WE CHANGE SOME VALUES IN THE WEIGHTAGE FUNCTION BELOW
        weight[0]=2 #MONTH_YEAR
####        weight[1]=0 #CITY NAME
        weight[2]=2 #CITY GROUP -- [Big Cities, Others]
        weight[3]=2 # TYPE OF RESTAURANT
        
        #MAKING THINGS FAST
##        for i in range (0,len(sdr)):
##            sdr[i]=sdr[i]* weight[i]
        sdr[0]=sdr[0]* weight[0]
        sdr[2]=sdr[2]* weight[2]
        sdr[3]=sdr[3]* weight[3]

        
        return(sdr)

    def KR_TESTING ():
        sdr_output_columns=300
        numOfOnesInBin=2
        noOfInputsInCityGroup=2
        noOfInputsInType=4 # THOUGH THERE ARE NO MB TYPES IN TRAIN BUT THERE ARE 291 MB TYPES IN TEST 
        noOfInputsInP1_P37=25
        percentage_of_on_bits_sdr=5
        no_of_connections_per_Neuron=39 #VERY IMPORTANT.
        input_weightage=numpy.ones((no_of_connections_per_Neuron)) #IF ANY PARTICULAR INPUT IS IMPORTANT WE INCREASE ITS WEIGHTAGE. SO CHANGE IN THAT INPUT DEFINETLY AFFECTS THE OUTPUT.
        
        on_bits_sdr=math.floor((percentage_of_on_bits_sdr) * (sdr_output_columns/100))
        cm1_CityGroup=numpy.random.rand(sdr_output_columns,(numOfOnesInBin*noOfInputsInCityGroup)) # USED TO CONVERT BINARY TO SDR. 2 MUTUALLY EXCLUSIVE INPUTS. SO 10 BIT REPRESNTATION
        cm1_Type=numpy.random.rand(sdr_output_columns,(numOfOnesInBin*noOfInputsInType)) # 3 MUTUALLY EXCLUSIVE INPUTS. SO 15 BIT REPRESNTATION
        cm1_p1_p37=numpy.random.rand(sdr_output_columns,(numOfOnesInBin+noOfInputsInP1_P37)) # 25 TOTAL DISTINCT INPUTS IN P1-P37
        cm2=CLASSIFICATION_SUPPORT.KP_Connection_Mat_2(sdr_output_columns,no_of_connections_per_Neuron) # ITS NOT A REGULAR cm. sdr_output_columns HAS TO BE > no_of_connections_per_Neuron
        x=[]
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
        list_of_list=[]

        
        f=open(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\train.csv','rt') #C:\Users\ahegshetye\Downloads\clt\kag\restaurnt\train.csv
        reader=csv.reader(f)
        for row in reader:
            x.append(row)


        for excel_rows in range(1,len(x)): #0TH INDEX HAS TITLES
            ip=[]
            sdr_rep=[]
            #CITY GROUP
            if x[excel_rows][3]=='Big Cities':
                ip.append(1)
            elif x[excel_rows][3]=='Other':
                ip.append(2)
            else:
                print(excel_rows)
                print(x[excel_rows][3])
                input(' 3rd unexpected city group found. ExCEL IS OVER. LETS BREAK')
                break


            #TYPE -  FC: Food Court=1, IL: Inline=2, DT: Drive Thru=3, MB: Mobile=4
            if x[excel_rows][4]=='FC':
                ip.append(1)
            elif x[excel_rows][4]=='IL':
                ip.append(2)
            elif x[excel_rows][4]=='DT':
                ip.append(3)
            elif x[excel_rows][4]=='MB':
                ip.append(4)
            else:
                input('unexpected TYPE ')


            #PA1 - PA37
            for i in range(5,42):
                ip.append(math.ceil(float(x[excel_rows][i])))
                



            #REVENUE
            ip.append(math.floor(int(x[excel_rows][42])/1000000) )

            #print(ip)
            #input(' IP LOOKS GOOD? ')

    ##      #CITY GROUP

            #MAX POSSIBLE INP=2 BUT TO KEEP ALL BIN REP SAME SIZE WE WILL TAKE THE TOTAL NUMBER OF POSSIBLE INPUTS,
            #NO OF ONES IN REP=2,MUTUALLY EXCLUSIVE .IT SHOULD GIVE 0 IF INPUT IS 0
            bin_rep1=(CLASSIFICATION_SUPPORT.Health_Converting_input_2binary(noOfInputsInCityGroup,numOfOnesInBin,ip[0],0)) 
            sdr_rep.append(CLASSIFICATION_SUPPORT.Binary2Sdr(bin_rep1,cm1_CityGroup,on_bits_sdr))


    ##      #TYPE
            bin_rep2=(CLASSIFICATION_SUPPORT.Health_Converting_input_2binary(noOfInputsInType,numOfOnesInBin,ip[1],0)) 
            sdr_rep.append(CLASSIFICATION_SUPPORT.Binary2Sdr(bin_rep2,cm1_Type,on_bits_sdr))


    ##      #P1 - P37
            #print(ip[2:39]) #ip[39] HAS THE CLASS VALUE
            for i in range(2,39):
                bin_rep3=(CLASSIFICATION_SUPPORT.Health_Converting_input_2binary(noOfInputsInP1_P37,numOfOnesInBin,ip[i],1))  #OVERLAPPING BITS
                sdr_rep.append(CLASSIFICATION_SUPPORT.Binary2Sdr(bin_rep3,cm1_p1_p37,on_bits_sdr))
            

            break
        return(input_weightage,sdr_rep)
##            print(input_weightage)
##            print(sdr_rep*input_weightage)

##            final_rep=CLASSIFICATION_SUPPORT.Health_Value_Mat(cm2,sdr_output_columns,on_bits_sdr,sdr_rep)
##            #input(final_rep)
##            
##
##
##            index=CLASSIFICATION_SUPPORT.KP_Extract_Indexes_ofOnes(on_bits_sdr,sdr_output_columns,final_rep) # GIVES INDEXES OF ALL 1'S
##            
##            CLASSIFICATION_SUPPORT.KR_Classifying(l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18,l19,l20,l21,l22,l23,l24,l25,ip[39],index) #ip[39] HAS THE CLASS VALUE







        

    #SPECIALITY OF KR_Value_Mat IS WE CAN PROVIDE WEIGHTAGE FOR EVERY INPUT. IF ANY INPUT HAS HIGH WEIGHT , CHANGE IN THAT INPUT WILL DEFINATLY CAUSE DRASTIC CHANGES IN THE OUTPUT . YAY
    def KR_Value_Mat(connection_matrix_2,sdr_output_columns,on_bits_sdr,sdr_list_1): #connections are added and progressed further
        #sdr_list_1= LIST OF SDR REPRESENTATION OF ALL SYMPTOMS. WHETHER YES OR NO DEPENDING ON INPUT.
        output=numpy.zeros((sdr_output_columns))# FINAL OUTPUT
        vm= numpy.zeros((sdr_output_columns,sdr_output_columns))
        size_sdr_list=len(sdr_list_1)
        for row in range(0,size_sdr_list): #OUTPUT CHANGES FOR EVERY INPUT.WE WILL ONLY CONSIDER THE FINAL ONE FOR CLASSIFICATION
            vm=vm+sdr_list_1[row] #input added
            #print('PROGRESSING INPUTS')
            output=CLASSIFICATION_SUPPORT.Health_Progressing_Inputs(connection_matrix_2,vm,sdr_output_columns,row,size_sdr_list,(row+1))# WE ARE JUST MOVING THE REQUIRED BIT FROM PREVIOUS CONNECTION TO NEXT CONNECTION. 
            #print('PROGRESSING INPUTS DONE!!')
        output=SENTENCE.top_bit(output,on_bits_sdr)
        return(output)
    
    def KR_Classifying(l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18,l19,l20,l21,l22,l23,l24,l25,l26,l27,l28,l29,l30,l31,l32,l33,l34,l35,l36,l37,l38,l39,l40,l41,l42,l43,l44,l45,l46,l47,l48,l49,l50,l51,l52,l53,l54,l55,l56,l57,l58,l59,l60,l61,l62,l63,l64,l65,l66,l67,l68,l69,l70,l71,l72,l73,l74,l75,l76,l77,l78,l79,l80,l81,l82,l83,l84,l85,l86,l87,l88,l89,l90,l91,l92,l93,l94,l95,l96,l97,l98,l99,l100,l101,l102,l103,l104,l105,l106,l107,l108,l109,l110,l111,l112,l113,l114,l115,l116,l117,
l118,l119,l120,l121,l122,l123,l124,l125,l126,l127,l128,l129,l130,l131,l132,l133,l134,l135,l136,l137,l138,l139,classification_vec,index):
         #DOEST MATTER IF THE index IS LIST OR ARRAY.. HURRAY !!
        if classification_vec==1:
            l1.append(index)
        elif classification_vec==2:
            l2.append(index)
        elif classification_vec==3:
            l3.append(index)
        elif classification_vec==4:
            l4.append(index)
        elif classification_vec==5:
            l5.append(index)
        elif classification_vec==6:
            l6.append(index)
        elif classification_vec==7:
            l7.append(index)
        elif classification_vec==8:
            l8.append(index)
        elif classification_vec==9:
            l9.append(index)
        elif classification_vec==10:
            l10.append(index)
        elif classification_vec==11:
            l11.append(index)
        elif classification_vec==12:
            l12.append(index)
        elif classification_vec==13:
            l13.append(index)
        elif classification_vec==14:
            l14.append(index)
        elif classification_vec==15:
            l15.append(index)
        elif classification_vec==16:
            l16.append(index)
        elif classification_vec==17:
            l17.append(index)
        elif classification_vec==18:
            l18.append(index)
        elif classification_vec==19:
            l19.append(index)
        elif classification_vec==20:
            l20.append(index)
        elif classification_vec==21:
            l21.append(index)
        elif classification_vec==22:
            l22.append(index)
        elif classification_vec==23:
            l23.append(index)
        elif classification_vec==24:
            l24.append(index)
        elif classification_vec==25:
            l25.append(index)
        elif classification_vec==26:
            l26.append(index)
        elif classification_vec==27:
            l27.append(index)
        elif classification_vec==28:
            l28.append(index)
        elif classification_vec==29:
            l29.append(index)
        elif classification_vec==30:
            l30.append(index)
        elif classification_vec==31:
            l31.append(index)
        elif classification_vec==32:
            l32.append(index)
        elif classification_vec==33:
            l33.append(index)
        elif classification_vec==34:
            l34.append(index)
        elif classification_vec==35:
            l35.append(index)
        elif classification_vec==36:
            l36.append(index)
        elif classification_vec==37:
            l37.append(index)
        elif classification_vec==38:
            l38.append(index)
        elif classification_vec==39:
            l39.append(index)
        elif classification_vec==40:
            l40.append(index)
        elif classification_vec==41:
            l41.append(index)
        elif classification_vec==42:
            l42.append(index)
        elif classification_vec==43:
            l43.append(index)
        elif classification_vec==44:
            l44.append(index)
        elif classification_vec==45:
            l45.append(index)
        elif classification_vec==46:
            l46.append(index)
        elif classification_vec==47:
            l47.append(index)
        elif classification_vec==48:
            l48.append(index)
        elif classification_vec==49:
            l49.append(index)
        elif classification_vec==50:
            l50.append(index)
        elif classification_vec==51:
            l51.append(index)
        elif classification_vec==52:
            l52.append(index)
        elif classification_vec==53:
            l53.append(index)
        elif classification_vec==54:
            l54.append(index)
        elif classification_vec==55:
            l55.append(index)
        elif classification_vec==56:
            l56.append(index)
        elif classification_vec==57:
            l57.append(index)
        elif classification_vec==58:
            l58.append(index)
        elif classification_vec==59:
            l59.append(index)
        elif classification_vec==60:
            l60.append(index)
        elif classification_vec==61:
            l61.append(index)
        elif classification_vec==62:
            l62.append(index)
        elif classification_vec==63:
            l63.append(index)
        elif classification_vec==64:
            l64.append(index)
        elif classification_vec==65:
            l65.append(index)
        elif classification_vec==66:
            l66.append(index)
        elif classification_vec==67:
            l67.append(index)
        elif classification_vec==68:
            l68.append(index)
        elif classification_vec==69:
            l69.append(index)
        elif classification_vec==70:
            l70.append(index)
        elif classification_vec==71:
            l71.append(index)
        elif classification_vec==72:
            l72.append(index)
        elif classification_vec==73:
            l73.append(index)
        elif classification_vec==74:
            l74.append(index)
        elif classification_vec==75:
            l75.append(index)
        elif classification_vec==76:
            l76.append(index)
        elif classification_vec==77:
            l77.append(index)
        elif classification_vec==78:
            l78.append(index)
        elif classification_vec==79:
            l79.append(index)
        elif classification_vec==80:
            l80.append(index)
        elif classification_vec==81:
            l81.append(index)
        elif classification_vec==82:
            l82.append(index)
        elif classification_vec==83:
            l83.append(index)
        elif classification_vec==84:
            l84.append(index)
        elif classification_vec==85:
            l85.append(index)
        elif classification_vec==86:
            l86.append(index)
        elif classification_vec==87:
            l87.append(index)
        elif classification_vec==88:
            l88.append(index)
        elif classification_vec==89:
            l89.append(index)
        elif classification_vec==90:
            l90.append(index)
        elif classification_vec==91:
            l91.append(index)
        elif classification_vec==92:
            l92.append(index)
        elif classification_vec==93:
            l93.append(index)
        elif classification_vec==94:
            l94.append(index)
        elif classification_vec==95:
            l95.append(index)
        elif classification_vec==96:
            l96.append(index)
        elif classification_vec==97:
            l97.append(index)
        elif classification_vec==98:
            l98.append(index)
        elif classification_vec==99:
            l99.append(index)
        elif classification_vec==100:
            l100.append(index)
        elif classification_vec==101:
            l101.append(index)
        elif classification_vec==102:
            l102.append(index)
        elif classification_vec==103:
            l103.append(index)
        elif classification_vec==104:
            l104.append(index)
        elif classification_vec==105:
            l105.append(index)
        elif classification_vec==106:
            l106.append(index)
        elif classification_vec==107:
            l107.append(index)
        elif classification_vec==108:
            l108.append(index)
        elif classification_vec==109:
            l109.append(index)
        elif classification_vec==110:
            l110.append(index)
        elif classification_vec==111:
            l111.append(index)
        elif classification_vec==112:
            l112.append(index)
        elif classification_vec==113:
            l113.append(index)
        elif classification_vec==114:
            l114.append(index)
        elif classification_vec==115:
            l115.append(index)
        elif classification_vec==116:
            l116.append(index)
        elif classification_vec==117:
            l117.append(index)
        elif classification_vec==118:
            l118.append(index)
        elif classification_vec==119:
            l119.append(index)
        elif classification_vec==120:
            l120.append(index)
        elif classification_vec==121:
            l121.append(index)
        elif classification_vec==122:
            l122.append(index)
        elif classification_vec==123:
            l123.append(index)
        elif classification_vec==124:
            l124.append(index)
        elif classification_vec==125:
            l125.append(index)
        elif classification_vec==126:
            l126.append(index)
        elif classification_vec==127:
            l127.append(index)
        elif classification_vec==128:
            l128.append(index)
        elif classification_vec==129:
            l129.append(index)
        elif classification_vec==130:
            l130.append(index)
        elif classification_vec==131:
            l131.append(index)
        elif classification_vec==132:
            l132.append(index)
        elif classification_vec==133:
            l133.append(index)
        elif classification_vec==134:
            l134.append(index)
        elif classification_vec==135:
            l135.append(index)
        elif classification_vec==136:
            l136.append(index)
        elif classification_vec==137:
            l137.append(index)
        elif classification_vec==138:
            l138.append(index)
        elif classification_vec==139:
            l139.append(index)
##        elif classification_vec==13:
##            l1.append(index)
        


    def KR_Converting_Lists2_ListOfList(l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18,l19,l20,l21,l22,l23,l24,l25,l26,l27,l28,l29,l30,l31,l32,l33,l34,l35,l36,l37,l38,l39,l40,l41,l42,l43,l44,l45,l46,l47,l48,l49,l50,l51,l52,l53,l54,l55,l56,l57,l58,l59,l60,l61,l62,l63,l64,l65,l66,l67,l68,l69,l70,l71,l72,l73,l74,l75,l76,l77,l78,l79,l80,l81,l82,l83,l84,l85,l86,l87,l88,l89,l90,l91,l92,l93,l94,l95,l96,l97,l98,l99,l100,l101,l102,l103,l104,l105,l106,l107,l108,l109,l110,l111,l112,l113,l114,l115,l116,l117,
l118,l119,l120,l121,l122,l123,l124,l125,l126,l127,l128,l129,l130,l131,l132,l133,l134,l135,l136,l137,l138,l139,list_of_list):
        #PUTS ALL L1,L2...L124 INTO LIST_OF_LIST
        
        list_of_list.append(l1)
        list_of_list.append(l2)
        list_of_list.append(l3)
        list_of_list.append(l4)
        list_of_list.append(l5)
        list_of_list.append(l6)
        list_of_list.append(l7)
        list_of_list.append(l8)
        list_of_list.append(l9)
        list_of_list.append(l10)
        list_of_list.append(l11)
        list_of_list.append(l12)
        list_of_list.append(l13)
        list_of_list.append(l14)
        list_of_list.append(l15)
        list_of_list.append(l16)
        list_of_list.append(l17)
        list_of_list.append(l18)
        list_of_list.append(l19)
        list_of_list.append(l20)
        list_of_list.append(l21)
        list_of_list.append(l22)
        list_of_list.append(l23)
        list_of_list.append(l24)
        list_of_list.append(l25)
        list_of_list.append(l26)
        list_of_list.append(l27)
        list_of_list.append(l28)
        list_of_list.append(l29)
        list_of_list.append(l30)
        list_of_list.append(l31)
        list_of_list.append(l32)
        list_of_list.append(l33)
        list_of_list.append(l34)
        list_of_list.append(l35)
        list_of_list.append(l36)
        list_of_list.append(l37)
        list_of_list.append(l38)
        list_of_list.append(l39)
        list_of_list.append(l40)
        list_of_list.append(l41)
        list_of_list.append(l42)
        list_of_list.append(l43)
        list_of_list.append(l44)
        list_of_list.append(l45)
        list_of_list.append(l46)
        list_of_list.append(l47)
        list_of_list.append(l48)
        list_of_list.append(l49)
        list_of_list.append(l50)
        list_of_list.append(l51)
        list_of_list.append(l52)
        list_of_list.append(l53)
        list_of_list.append(l54)
        list_of_list.append(l55)
        list_of_list.append(l56)
        list_of_list.append(l57)
        list_of_list.append(l58)
        list_of_list.append(l59)
        list_of_list.append(l60)
        list_of_list.append(l61)
        list_of_list.append(l62)
        list_of_list.append(l63)
        list_of_list.append(l64)
        list_of_list.append(l65)
        list_of_list.append(l66)
        list_of_list.append(l67)
        list_of_list.append(l68)
        list_of_list.append(l69)
        list_of_list.append(l70)
        list_of_list.append(l71)
        list_of_list.append(l72)
        list_of_list.append(l73)
        list_of_list.append(l74)
        list_of_list.append(l75)
        list_of_list.append(l76)
        list_of_list.append(l77)
        list_of_list.append(l78)
        list_of_list.append(l79)
        list_of_list.append(l80)
        list_of_list.append(l81)
        list_of_list.append(l82)
        list_of_list.append(l83)
        list_of_list.append(l84)
        list_of_list.append(l85)
        list_of_list.append(l86)
        list_of_list.append(l87)
        list_of_list.append(l88)
        list_of_list.append(l89)
        list_of_list.append(l90)
        list_of_list.append(l91)
        list_of_list.append(l92)
        list_of_list.append(l93)
        list_of_list.append(l94)
        list_of_list.append(l95)
        list_of_list.append(l96)
        list_of_list.append(l97)
        list_of_list.append(l98)
        list_of_list.append(l99)
        list_of_list.append(l100)
        list_of_list.append(l101)
        list_of_list.append(l102)
        list_of_list.append(l103)
        list_of_list.append(l104)
        list_of_list.append(l105)
        list_of_list.append(l106)
        list_of_list.append(l107)
        list_of_list.append(l108)
        list_of_list.append(l109)
        list_of_list.append(l110)
        list_of_list.append(l111)
        list_of_list.append(l112)
        list_of_list.append(l113)
        list_of_list.append(l114)
        list_of_list.append(l115)
        list_of_list.append(l116)
        list_of_list.append(l117)
        list_of_list.append(l118)
        list_of_list.append(l119)
        list_of_list.append(l120)
        list_of_list.append(l121)
        list_of_list.append(l122)
        list_of_list.append(l123)
        list_of_list.append(l124)
        list_of_list.append(l125)
        list_of_list.append(l126)
        list_of_list.append(l127)
        list_of_list.append(l128)
        list_of_list.append(l129)
        list_of_list.append(l130)
        list_of_list.append(l131)
        list_of_list.append(l132)
        list_of_list.append(l133)
        list_of_list.append(l134)
        list_of_list.append(l135)
        list_of_list.append(l136)
        list_of_list.append(l137)
        list_of_list.append(l138)
        list_of_list.append(l139)

######----------------------------------------------------------------------------------------------------------------------------
                            #HEALTH
    def Conv_inp_2Rep (inp,NumberOfSymptoms): #CONVERTS INPUT TO A REPRESENTATION. IF INPUT IS (1,2,3) AND NumberOfSymptoms=5 THEN REPRESENTATION=(1,2,3,0,0)
    #IF SYMPTOM EXIST IT HAS A NUMBER ELSE 0. SIMPLE AS SHIT. NO COMPLEX ODD EVEN BUSINESS FUCCKER
        rep=numpy.zeros((NumberOfSymptoms))
        for i in range (0,len(inp)):
            rep[inp[i]-1]=inp[i] #SINCE INDEXING STARTS FROM 0
        return (rep)
        
    
    def Array_ofOddEvenNumbers(x,y,NumberOfSymptoms): # x=1 then odd , x=2 then even numbers. y is VECTOR. FUNCTION PUTS ALL EVEN OR ODD NUMBERS  (NO'S)  IN THAT VECTOR. AND LATER ALL SYMPTOMS ARE CONVERTED TO YES'S
        Ysiz=len(y)
        output=numpy.zeros((NumberOfSymptoms))
        
        #SETTING THE OUTPUT TO NO'S.. SO SYMPTOMS DONT EXIST.. LIKE NUMPY.ZEROS 
        if x==1:
            indx=0
            for i in range(1,(NumberOfSymptoms*2),2): # ODD.. WHOLE Ysiz VECTOR NOW HAS ODD NUMBERS ON ITS INDEXES
                output[indx]=i
                indx=indx+1
        if x==2:
            indx=0
            for i in range(2,((NumberOfSymptoms*2)+2),2): # EVEN
                output[indx]=i
                indx=indx+1
        

        
        #NOW REPLACING INPUT SYMPTOMS INDEXES FROM NO'S TO YES. THOSE SYMPTOMS WERE SEEN IN THIS TEST CASE. SO THOSE WILL BE YES AND OTHERS WILL CONTINUE TO REMAIN 'NO'
        for indx2 in range (0,Ysiz): #EVERY SYMPTOM AT A TIME
            
            output[y[indx2]-1]=output[y[indx2]-1]-1  #INDEX IN PYTHON START FROM 0 AND OUR SYMPTOMS START FROM 1. SO WE SUBSTRACT 1 FROM THE INDEX. THEN WE DO -1 ON THE WHOLE VALUE TO CONVERT IT FROM EVEN "NO" TO ODD "YES"

        return(output)




    def Health_Value_Mat(connection_matrix_2,sdr_output_columns,on_bits_sdr,sdr_list_1): #connections are added and progressed further
        #sdr_list_1= LIST OF SDR REPRESENTATION OF ALL SYMPTOMS. WHETHER YES OR NO DEPENDING ON INPUT.
        #IT WILL GIVE OUPUT ONLY WHEN LAST SDR [sdr_list_1] INPUT IS PROCESSED. ALL OTHER INPUTS DONT GENERATE A OUTPUT, THEY JUST KEEP ADDING VALUE TO THE FINAL OUTPUT.
        output=numpy.zeros((sdr_output_columns))# FINAL OUTPUT
        vm= numpy.zeros((sdr_output_columns,sdr_output_columns))
        size_sdr_list=len(sdr_list_1)
        for row in range(0,size_sdr_list): #OUTPUT CHANGES FOR EVERY INPUT.WE WILL ONLY CONSIDER THE FINAL ONE FOR CLASSIFICATION
            vm=vm+sdr_list_1[row] #input added
            #print('PROGRESSING INPUTS')
            output=CLASSIFICATION_SUPPORT.Health_Progressing_Inputs(connection_matrix_2,vm,sdr_output_columns,row,size_sdr_list,(row+1))# WE ARE JUST MOVING THE REQUIRED BIT FROM PREVIOUS CONNECTION TO NEXT CONNECTION. 
            #print('PROGRESSING INPUTS DONE!!')
        output=SENTENCE.top_bit(output,on_bits_sdr)
        return(output)




    def Health_Progressing_Inputs(cm,vm,sdr_output_columns,input_number,total_inputs,destination_input):#NEW EFFICIENT SHIFTING OF connections !!!
        #input_number = THE ROW OF SDR MAT 1. ITS THE NUMBER OF INPUT. 1ST INPUT OR 2ND OR 3RD ETC.
        #total_inputs = TOTAL NUMBER OF ROWS IN SDR MAT 1. TOTAL INPUTS TO BE PROCESED FOR FINAL OUTPUT.
        #destination_input= FINAL CONNECTION WHERE WE WANT TO MOVE THE CURRENT CONNECTION OF A NEURON. GENERALLY WE MOVE 1ST CONECTN TO 2ND CONNECTN, 2ND TO 3RD, NTH TO N+1.

        vm2=vm*0#  TEMPORARY VALUE MATRIX
        output=numpy.zeros((sdr_output_columns))# FINAL OUTPUT
        for row in range (0,sdr_output_columns): #NUMBER OF NEURONS
            if (input_number<total_inputs-1): # IF SDR INPUT IS NOT THE LAST INPUT OF THE SEQUENCE THEN ITS NOT ASSIGNED TO FINAL OUTPUT.IT JUST SHIFTS IN THE vm
                vm[row,cm[row,destination_input]]=vm[row,cm[row,input_number]]
                vm[row,cm[row,input_number]]=0
            elif (input_number==total_inputs-1):
                output[row]=vm[row,cm[row,input_number]]
        return(output)


    ''' UNIVERSAL  .. JUST TAKING THE REQUIRED CONNECTIONS AS WE ARE ONLY TAKING 1 OUTPUT AT THE END OF ALL INPUTS PROCESSING.
    if (input_number<total_inputs-1):
    vm[row,cm[row,destination_input]]=vm[row,cm[row,input_number]]
    vm[row,cm[row,input_number]]=0
    elif (input_number==total_inputs-1)
    output[row]=vm[row,cm[row,input_number]]
    return(output)
    '''


    def Converting_Lists2_ListOfList(l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18,l19,l20,l21,l22,l23,l24,l25,l26,l27,l28,l29,l30,l31,l32,l33,l34,l35,l36,l37,l38,l39,l40,l41,l42,l43,l44,l45,l46,l47,l48,l49,l50,l51,l52,l53,l54,l55,l56,l57,l58,l59,l60,l61,l62,l63,l64,l65,l66,l67,l68,l69,l70,l71,l72,l73,l74,l75,l76,l77,l78,l79,l80,l81,l82,l83,l84,l85,l86,l87,l88,l89,l90,l91,l92,l93,l94,l95,l96,l97,l98,l99,l100,l101,l102,l103,l104,l105,l106,l107,l108,l109,l110,l111,l112,l113,l114,l115,l116,l117,
l118,l119,l120,l121,l122,l123,l124,list_of_list):
        #PUTS ALL L1,L2...L124 INTO LIST_OF_LIST
        list_of_list.append(l1)
        list_of_list.append(l2)
        list_of_list.append(l3)
        list_of_list.append(l4)
        list_of_list.append(l5)
        list_of_list.append(l6)
        list_of_list.append(l7)
        list_of_list.append(l8)
        list_of_list.append(l9)
        list_of_list.append(l10)
        list_of_list.append(l11)
        list_of_list.append(l12)
        list_of_list.append(l13)
        list_of_list.append(l14)
        list_of_list.append(l15)
        list_of_list.append(l16)
        list_of_list.append(l17)
        list_of_list.append(l18)
        list_of_list.append(l19)
        list_of_list.append(l20)
        list_of_list.append(l21)
        list_of_list.append(l22)
        list_of_list.append(l23)
        list_of_list.append(l24)
        list_of_list.append(l25)
        list_of_list.append(l26)
        list_of_list.append(l27)
        list_of_list.append(l28)
        list_of_list.append(l29)
        list_of_list.append(l30)
        list_of_list.append(l31)
        list_of_list.append(l32)
        list_of_list.append(l33)
        list_of_list.append(l34)
        list_of_list.append(l35)
        list_of_list.append(l36)
        list_of_list.append(l37)
        list_of_list.append(l38)
        list_of_list.append(l39)
        list_of_list.append(l40)
        list_of_list.append(l41)
        list_of_list.append(l42)
        list_of_list.append(l43)
        list_of_list.append(l44)
        list_of_list.append(l45)
        list_of_list.append(l46)
        list_of_list.append(l47)
        list_of_list.append(l48)
        list_of_list.append(l49)
        list_of_list.append(l50)
        list_of_list.append(l51)
        list_of_list.append(l52)
        list_of_list.append(l53)
        list_of_list.append(l54)
        list_of_list.append(l55)
        list_of_list.append(l56)
        list_of_list.append(l57)
        list_of_list.append(l58)
        list_of_list.append(l59)
        list_of_list.append(l60)
        list_of_list.append(l61)
        list_of_list.append(l62)
        list_of_list.append(l63)
        list_of_list.append(l64)
        list_of_list.append(l65)
        list_of_list.append(l66)
        list_of_list.append(l67)
        list_of_list.append(l68)
        list_of_list.append(l69)
        list_of_list.append(l70)
        list_of_list.append(l71)
        list_of_list.append(l72)
        list_of_list.append(l73)
        list_of_list.append(l74)
        list_of_list.append(l75)
        list_of_list.append(l76)
        list_of_list.append(l77)
        list_of_list.append(l78)
        list_of_list.append(l79)
        list_of_list.append(l80)
        list_of_list.append(l81)
        list_of_list.append(l82)
        list_of_list.append(l83)
        list_of_list.append(l84)
        list_of_list.append(l85)
        list_of_list.append(l86)
        list_of_list.append(l87)
        list_of_list.append(l88)
        list_of_list.append(l89)
        list_of_list.append(l90)
        list_of_list.append(l91)
        list_of_list.append(l92)
        list_of_list.append(l93)
        list_of_list.append(l94)
        list_of_list.append(l95)
        list_of_list.append(l96)
        list_of_list.append(l97)
        list_of_list.append(l98)
        list_of_list.append(l99)
        list_of_list.append(l100)
        list_of_list.append(l101)
        list_of_list.append(l102)
        list_of_list.append(l103)
        list_of_list.append(l104)
        list_of_list.append(l105)
        list_of_list.append(l106)
        list_of_list.append(l107)
        list_of_list.append(l108)
        list_of_list.append(l109)
        list_of_list.append(l110)
        list_of_list.append(l111)
        list_of_list.append(l112)
        list_of_list.append(l113)
        list_of_list.append(l114)
        list_of_list.append(l115)
        list_of_list.append(l116)
        list_of_list.append(l117)
        list_of_list.append(l118)
        list_of_list.append(l119)
        list_of_list.append(l120)
        list_of_list.append(l121)
        list_of_list.append(l122)
        list_of_list.append(l123)
        list_of_list.append(l124)


    def Health_Converting_input_2binary(n,m,k,exc_overlap): # CONVERTING INPUT (1,2,3,...) TO A BINARY REPRESENTATION WHICH IS FURTHER CONVERTED TO SDR
        #I HAVE ADDED EXTRA ZERO CONDITION. IF INPUT IS 0 THEN  RETURN ALL ZEROS
        # n- TOTAL NUMBER OF DISTINCT INPUTS IN THE PROGRAM
        # m-NUMBER OF 1'S IN EVERY BINARY REPRESENTATION NEEDED
        # k- WHICH INPUT (1ST,2ND,3RD..) NO ZEROS PLEASE !!!!
        #exc_overlap- IF 0 THEN MUTUALLY EXCLUSIVE OUTPUT NEEDED. IF 1 THEN OVERLAPING OUTPUT NEEDED
        if exc_overlap ==0:
            output=numpy.zeros((m*n)) #MUTUALY EXCLUSIVE NEEDS BIGGER SIZE.
        elif exc_overlap==1:
            output=numpy.zeros((m+n)) #OVERLAPPING NEEDS SMALLER SIZE.
            
        if k!=0: #IF K (NPUT) = 0 THEN WE JUST RETURN 0. NO BIN REPRENTATION NEEDED.
            
            if exc_overlap ==0:
                for i in range (1,m+1):
                    output[(k*m)-i]=1
            elif exc_overlap ==1: #OVERLAPPING. WE NEED OVERLAPPING BITS FOR ANALOG SIGNALS WHERE TWO CONSECUTING INPUTS NEED OVERLAPING BITS TO PRESERVE SOME SEMANTIC SIMILARITY. BUT AS THE DIFFERENCE BETWEEN INPUTS INCREASE THE SIMILARITY SHOULD REDUCE AND EVENTUALLY VANISH
                #for i in range (0,m):
                if (k-1+(m-1))>=(m+n): # NOW IF THE INPUT IS BIGGER THAN OUR REPRESENTATION CAN REPRESENT, IT WILL JUST GIVE THE MAX AMOUNT WITHOUT CAUSING ERRORS.
                    output[m+n-m:m+n]=1
                else:
                    output[k-1:k-1+m]=1
        return output
######----------------------------------------------------------------------------------------------------------------------------



    def KP_temp_writingTocsv(a):
        resultFile = open(r"C:\Users\Aseem\Documents\KAGGLE\Marine plankton\Final_Output.csv",'w',newline='') # TO AVOID NEW LINES AFTER EVERY WRIT IN EXCEL
        wr = csv.writer(resultFile, dialect='excel')
        wr.writerows(a)
        resultFile.close()


    def KP_CombineImageNames_Probabilities_write2CSV():
        fin=open(r'C:\Users\Aseem\Documents\KAGGLE\Marine plankton\Test_Images_Names.txt','r')
        
        #   WE COPY PASTED IT DIRECTLY IN EXCEL :d
##        while(True):
##            try:
##                image_names.append((fin.readline()).replace('\n',''))
##            except ValueError: # MEANS ALL NAMES ARE OVER NOW. SO BREAK.
##                break

        prob_ofClasses_forAll_Images=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Marine plankton\prob_ofClasses_forAll_Images.npy')
        CLASSIFICATION_SUPPORT.KP_temp_writingTocsv(prob_ofClasses_forAll_Images.tolist())
##        while (True):
##            try:
##                Final_result.append(image_names[indx+1]+prob_ofClasses_forAll_Images[indx])
##                indx=indx+1
##            except ValueError: # MEANS ALL NAMES ARE OVER NOW. SO BREAK.
##                break
##            
##
##        
##        resultFile = open("C:\Users\Aseem\Documents\KAGGLE\Marine plankton\Final_Output.csv",'wt')
##        wr = csv.writer(resultFile, dialect='excel')
##        wr.writerows(Final_result)







        

    def KP_Convert_indexVector_to_OnesVector(vec,leng_output): #TAKES VECTOR WITH INDEXED OF ONES. AND GIVES A VECTOR WITH ONES IN THOSE INDEXES. ALSO TAKES IN THE ACTUAL LENGTH OF VECTOR
        vec_op=numpy.zeros((leng_output))
        for i in range(0,len(vec)):
            vec_op[vec[i]]=1
        return(vec_op)
            

    def KP_Converting_MatlabTxt_npy (): #READS THE TEST IMAGE NAME MATLAB TEXT FILE AND CONVERTS IT TO NPY FOR MERGING LATER
        #prob_ofClasses_forAll_Images=numpy.load('C:\Users\Aseem\Documents\KAGGLE\Marine plankton\prob_ofClasses_forAll_Images.npy')
        fin=open(r'C:\Users\Aseem\Documents\KAGGLE\Marine plankton\Test_Images_Names.txt','r')
        l1=fin.readline()



    def KP_Predicting_Class_2(img,CM3_CLASSIFIER,no_of_classes,bits_per_class): # NEW TYPE OF PREDICTION WHERE WE MAKE IMAGE REPESENTATION GO THROUGH OUR SPECIAL CM3_CLASSIFIER AND THEN WE COMPARE THE OUTPUT WITH OUR EXISTING LEARNED MUTUALLY EXCLUSIVE CLASS REPRESENTATION
                                                            #bits_per_class = KP_on_bits_sdr
        #list_of_list_distinct=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Marine plankton\train\saved\list_of_list_distinct.npy')
        #img=CLASSIFICATION_SUPPORT.KP_Convert_indexVector_to_OnesVector(list_of_list_distinct[1][0],300)
        #CM3_CLASSIFIER=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Marine plankton\CM3_CLASSIFIER.npy')


        
        class_statistic=sum((img*CM3_CLASSIFIER).T) #ITS A VECTOR
        #CURRENTLY FINAL CLASS REPRESENTATION HAS 1ST CLASS REPRESENTED AT THE END. 2ND IN THE BEGINNING AND 3RD AFTER 2ND AND SO ON..EVERY CLASS HAS KP_on_bits_sdr NUMBER OF BITS IN class_statistic
        #WE NOW EXTRACT EVERY CLASS TOTAL MATCH AND PROVIDE A PROBABILITY.
        #prob_forAll_Images=[] # EVERY CLASS PROBABILITY FOR EVERY IMAGE
        prob_ofClasses_perImage=[] #121 CLASS PROBABILITIES FOR ONE IMAGE
        for i in range(0,no_of_classes):
            probability_of_class=(sum(class_statistic[(i*bits_per_class):((i*bits_per_class)+bits_per_class)])/bits_per_class)/bits_per_class #SINCE THRESHOLD IS 1. MAX VALUE OF ANY BIT WILL BE bits_per_class
            prob_ofClasses_perImage.append(format(round(probability_of_class,4))) #APPENDING PROBABLITY OF EVERY CLASS FOR ONE IMAGE. IT WILL HAVE 121 CLASS PROBABILITIES. .MAY BE FORMAT IS CONVERTING IT TO STRING. MAY BE.!!

        return(prob_ofClasses_perImage)
        






        
    #THIS FUNCTIONS FUCKS EVERYTHING UP. DOESNT RETAIN THE DISTINCTNESS OF EACH PATTERN.
    def KP_Class_Representation(KP_on_bits_sdr,KP_sdr_output_columns): #BUILD CM3_CLASSIFIER THAT WOULD CLASSIFY ANY IMAGE REP INTO ONE OF THE CLASS REPRESENTATION. CM3_CLASSIFIER BUILDS CONNECTIONS FOR EVERY IMAGE BELONGING TO A CLASS AND BOUNDS IT TO A CLASS REPRESENTATION.
                                    #HERE WE HAVE 121 CLASSES. SO IT WILL GENERATE 121 RANDOM MUTUALLY EXCLUSIVE REPRE AND LEARN CONNECTIONS FOR EVERY IMAGE OF A PARTICULAR CLASS TO GENERATE THAT REPRESENTATION.
                                    #AFTER LEARNING ALL IMAGES OF 1 CLASS IT WILL GENERATE SAME CLASS REPRESENTATION FOR ANY OF THOSE IMAGES WHEN THOSE IMAGES GO THROUGH CM3_CLASSIFIER
        list_of_list_distinct=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Marine plankton\train\saved\list_of_list_distinct.npy')
        len_distinct_list=len(list_of_list_distinct)
        final_class_rep=[] # ITS A LIST OF ALL 121 CLASSES. EVERY CLASS HAS 15 ON BITS. [111000..] [000111000...] LIKE THAT.
        for i in range(1,len_distinct_list+1): #FROM 1 BECAUZ IN Converting_input_2binary NO ZERO'S ARE ALLOWD . IT HAS TO BE 1ST INPUT OR 2ND. ETX IF WE DO FROM 0 TO LENTGH .. THEN 1ST REPRESENTATION WILL BE 1'S IN THE END. 2ND WILL BE 1ST IN BEGINING. ND THEN CONTINUE.
            final_class_rep.append(CLASSIFICATION_SUPPORT.Converting_input_2binary(len_distinct_list,KP_on_bits_sdr,i,0)) #STORING ALL NEW CLASS REPRESENTATIONS IN final_class_rep.NO OF BITS PER CLASS =KP_on_bits_sdr JUST TO KEEP IT UNIFORM

        length_final_classification=len(final_class_rep[0]) #LENTH OF FINAL CLASS REPRESENTATION . WE HAVE 121 REPRESENTATION FOR KP
        CM3_CLASSIFIER=numpy.zeros((length_final_classification,KP_sdr_output_columns))

        for classes in range(0,len_distinct_list):
            for img in range(0,len(list_of_list_distinct[classes])):
                img_rep=CLASSIFICATION_SUPPORT.KP_Convert_indexVector_to_OnesVector(list_of_list_distinct[classes][img],KP_sdr_output_columns) #CONVERTING AN INDEX OF 1S VECTOR TO ACTUAL VECTOR WITH ONES AT THOSE INDICES
                for bits in range(0,length_final_classification):
                    
                    if final_class_rep[classes][bits]>0: # POSITIVE BIT IN CLASS REPRESENTATION. STRENTHEN THE CONNECTIONS OF ON INPUT BITS
                       CM3_CLASSIFIER[bits,:]=CM3_CLASSIFIER[bits,:]+img_rep  #ADDING IMAGE REPRESENTATION TO THAT ROW. SO STRENGTHENINNG CONNECIONS. REFOR NOTEBOOK
                       CM3_CLASSIFIER[bits,:]=CLASSIFICATION_SUPPORT.thresholding_of_matrix_2(CM3_CLASSIFIER[bits,:],1) # WE CHOSE THRESHOLD TO BE 1 HERE. WE WANT TO LEARN IMAGES IRRESPECTIVE OF HOW MANY TIMES SAME IMAGE IS SHOWN.LATER IF SAME BIT COMES AGAIN IT WILL STILL HAVE CONNECTION STRNGTH 1 MAX.SO A GROUP OF BITS REPRESENT A PATTERN AND NOT JST ONE BIT
                                                                                               # MAX VALUE IS THRESHOLD. ALL BITS > THRESH ARE MADE = THRESH

        numpy.save(r'C:\Users\Aseem\Documents\KAGGLE\Marine plankton\CM3_CLASSIFIER.npy',CM3_CLASSIFIER)
        numpy.save(r'C:\Users\Aseem\Documents\KAGGLE\Marine plankton\final_class_rep.npy',final_class_rep)
        return(final_class_rep,CM3_CLASSIFIER)                                                                 
                       

        
            
        

    def Put_QuadReprntatn_in_SDR(q1,q2,q3,q4,sdr_mat):#PUTS 4 QUADRANT REPRESENTATIONS INTO 1 FINAL SDR_MATRIX
        i=0
        sdr_mat[i,:]=q1
        sdr_mat[i+1,:]=q2
        sdr_mat[i+2,:]=q3
        sdr_mat[i+3,:]=q4

    def KP_Conv_list_to_Matrix(value_list,length_list,sdr_mat,length_value_list,length_length_list): # CONVERTS TWO LISTS (VALUE+LENGTH OF A QUADRANT) INTO SDR MATRIX. SDR MAT IS ALREADY ALL SET TO ZEROS IN THE BEGINNING
        for val_indx in range (length_value_list):
            sdr_mat[val_indx,:]=value_list[val_indx]
        for  ln_indx in range (length_length_list): #FIRST 20 ROWS FOR VALUE AND NEXT 20 FOR LENGTH EACH QUADRANT
            sdr_mat[20+ln_indx,:]=length_list[ln_indx]
      #DONT HAVE TO RETURN SDR_MAT COZ IN PYTHON EVERYTHIN IS PASSED BY REFERENCE. SO ITS ORIGINAL VALUE IS ALREADY CHANGED


    def KP_Connection_Mat_2(sdr_output_columns,no_of_connections_per_Neuron):#THIS NEW CM2 STORES INDEXES OF CONNECTIONS. iNDEX OF CM2 ROW (0,1,2,3..) = NEURON CONNECTIONS NUMBER (1ST CONN, 2ND CONNEC,3RD,4TH..)
                            # VALUE OF CM2[8,1,11,5] = ACTUAL VALUE MAT INDEX.
                            #CM[[2,0,0,0,4,0,0,1,0,0,3],[4,0,0,3,0,2,0,0,0,0,1]] --  CM2[[8,1,11,5],[11,6,4,1] ..] THESE IS REAL EXAMPLE 
                            #refer notebook for more !!
        cm2= numpy.zeros((sdr_output_columns,no_of_connections_per_Neuron))
        for row in range(0,sdr_output_columns):
            cm2[row,:]=random.sample(range(sdr_output_columns), no_of_connections_per_Neuron)#random indexes [99, 36, 19, 82, 61, 10, 69, 33, 98, 37]
        return(cm2)

    def KP_Value_Mat(connection_matrix_2,sdr_output_columns,sdr_matrix_1,on_bits_sdr,no_of_connections_per_Neuron,KP_max_value_perQuadrnt,img_quad): #connections are added and progressed further
        #img_quad= 0 MEANS THIS FUNCTION IS BEING USED TO BUILD FINAL IMAGE REPRESENTATION.  1 MEANS ITS BEING USED TO BUILD QUADRANT REPRESENTATION
        output=numpy.zeros((sdr_output_columns))# FINAL OUTPUT
        vm= numpy.zeros((sdr_output_columns,sdr_output_columns))
        sdr_rows=sdr_matrix_1.shape[0]
        row=0
        while row< sdr_rows : #OUTPUT CHANGES FOR EVERY INPUT.WE WILL ONLY CONSIDER THE FINAL ONE FOR CLASSIFICATION
            #FOLLOWING PATCH IS JUST FOR KAGGLE PLANKTON. EVERY SDR MAT IS BUILT OF VALUE AND LENGTH. 1ST 20 IS VALUE AND 20-40 IS LENGTH.
            #IF THERE IS A ZERO ROW IN ANY [SECTION] THAT MEANS THE REMAINING SECTION IS GOING TO BE 0 TOO. SO TO PACE THINGS UP WE CAN JUST STOP PROGRESSING FOR REMAINING SECTION AND EITHER SKIP TO ANOTHER SECTION OR MAKE THE CURRENT OUTPUT AS FINAL OUTPUT DEPENDING OF WHETHER ITS 1ST SECTION 0 ROW OR 2ND RESPECTIVELY
##            if sum(sdr_matrix_1[row,:])==0:
##                if row< max_value_perQuadrnt-1: #max_value_perQuadrnt =20 FOR NOW. ONE SDR MAT WILL HAVE 40 ROWS. 20 VALUES AND 20 LENGTHS
##                    row=max_value_perQuadrnt-1
##                elif row> max_value_perQuadrnt-1:
##                    break # ALL FURTHER ROWS ARE GOING TO BE 0 SO NO POINT PROGRESSING OUTPUTS. CURRENT OUTPUT
            #BUT HERE WE WILL HAVE TO PICK UP OUTPUT FROM MIDDLE OF CONNECTIONS. 


            vm=vm+sdr_matrix_1[row,:] #input added

            
            '''
#TO MAKE IT UNIVERSAL FOR OTHER CODES TOO.. REMOVE THIS FOLLOWING BLOCK .. REMOVE DESTINATION_INPUT VARIABLE AND JUST MAKE IT PREOGRESS ONLY THE REQUIRED CONNECTION = INPUT NUMBER.
    vm=vm+sdr_matrix_1[row,:] #input added
    (output)=CLASSIFICATION_SUPPORT.KP_Progressing_Inputs(connection_matrix_2,vm,sdr_output_columns,no_of_connections_per_Neuron,row,sdr_rows)
    output=SENTENCE.top_bit(output,on_bits_sdr)
    return(output)
            '''
            
            if sum(sdr_matrix_1[row,:])==0 and (img_quad ==1): #ONLY SUITS QUADRANT REPRESENATTION
                #print('sum of sdr matrix row is 00. row ==', row,'sdr_rows = ',sdr_rows, 'img_quad = ', img_quad)
                
                if row<= KP_max_value_perQuadrnt-1: #KP_max_value_perQuadrnt =20 FOR NOW. ONE SDR MAT WILL HAVE 40 ROWS. 20 VALUES AND 20 LENGTHS
                    destination_input = KP_max_value_perQuadrnt #MAKING DESTINATION OF THE CONNECTION AS THE BEGINNING OF LENGTH IN THAT SDR MATRIX.THIS IS END OF VALUE PATTERNS AND START OF LENGTH PATTERNS IN SDR AND NEURON CONNECTIONS BOTH
                    (output)=CLASSIFICATION_SUPPORT.KP_Progressing_Inputs(connection_matrix_2,vm,sdr_output_columns,no_of_connections_per_Neuron,row,sdr_rows,destination_input)
                    row=KP_max_value_perQuadrnt #MAKING THE ROW TO POINT TO BEGINNING OF LENGTH PATTERNS IN SDR MATRIX
                    #print('KP_max_value_perQuadrnt = ',KP_max_value_perQuadrnt,'\n row is set to it. row = ', row)
                elif row> KP_max_value_perQuadrnt-1:  #MEANS LEGNTH PATTERNS ARE OVER TOO. DIRECTLY GIVE THE OUTPUT NOW. ENOUGH OF PROGRESSING NEURON CONNECTIONS
                    #print( '!!! yay row is greater than 19. we FOUND 0 IN LENGTH ')
                    destination_input = (sdr_rows)
                    (output)=CLASSIFICATION_SUPPORT.KP_Progressing_Inputs(connection_matrix_2,vm,sdr_output_columns,no_of_connections_per_Neuron,row,sdr_rows,destination_input)
                    row=sdr_rows -1
                    #print('row is set to 39 now coz sdr_rows = ', sdr_rows , 'row = ', row)
                    break

            else:
            
                #print('PROGRESSING INPUTS')
            
                #print('vm = \n' ,vm)
                (output)=CLASSIFICATION_SUPPORT.KP_Progressing_Inputs(connection_matrix_2,vm,sdr_output_columns,no_of_connections_per_Neuron,row,sdr_rows,row+1)
                #print('output =\n' ,output)
                row=row+1
            
            #input('PROGRESSING INPUTS DONE!!')
        output=SENTENCE.top_bit(output,on_bits_sdr)
        return(output)
        

    def KP_Progressing_Inputs(cm,vm,sdr_output_columns,no_of_connections_per_Neuron,input_number,total_inputs,destination_input):#NEW EFFICIENT SHIFTING OF connections !!!
        #input_number = THE ROW OF SDR MAT 1. ITS THE NUMBER OF INPUT. 1ST INPUT OR 2ND OR 3RD ETC.
        #total_inputs = TOTAL NUMBER OF ROWS IN SDR MAT 1. TOTAL INPUTS TO BE PROCESED FOR FINAL OUTPUT.
        #destination_input= FINAL CONNECTION WHERE WE WANT TO MOVE THE CURRENT CONNECTION OF A NEURON. GENERALLY WE MOVE 1ST CONECTN TO 2ND CONNECTN, 2ND TO 3RD, NTH TO N+1. BUT HERE IN KAGGLE ONCE A ROW IS ZERO MEANS ALL FUTURE ROWS FOR THAT PATTERN ARE GOING TO BE ZERO. SO WHY NOT DIRECTLY MOVE THE CONNECTION VALUE TO END CONNECTIION.

        #vm2=vm*0#  TEMPORARY VALUE MATRIX
        output=numpy.zeros((sdr_output_columns))# FINAL OUTPUT
        for row in range (0,sdr_output_columns): #NUMBER OF NEURONS
            
            #WE ARE ONLY CONSIDERING CONNECTION NUMBER= INPUT NUMBER. FOR FIRST INPUT WE CONSIDER 1ST CONNECTION OF EVERY NEURON. FOR 2ND INPUT 2ND CONNECTION. SO IN END FOR 40TH INPUT WE WIL CONSIDER 40TH CONNECTION WHICH WILL INTURN GIVE US THE FINAL OUTPUT.THIS WILL SAVE PROCESSING TIME!!
            #CONNECTIONS PER NEURON HAVE TO BE EQUAL TO TOTAL INPUTS IN A SEQUENCE(A QUADRANT HERE)
            if (input_number<total_inputs-1) and( destination_input != total_inputs) :# PYTHON INDEX STARTS FROM 0
                vm[row,cm[row,destination_input]]=vm[row,cm[row,input_number]]
                vm[row,cm[row,input_number]]=0
            elif (input_number==total_inputs-1)or ( destination_input == total_inputs) : #  MEANS WE HAD ZERO IN LENGTH PATTERNS OR WE REACHED END OF INPUT SDR MATR.JUST GIVE OUTPUT
                output[row]=vm[row,cm[row,input_number]]
        return(output)
    
    ''' UNIVERSAL  .. JUST TAKING THE REQUIRED CONNECTIONS AS WE ARE ONLY TAKING 1 OUTPUT AT THE END OF ALL INPUTS PROCESSING.
if (input_number<total_inputs-1):
                vm[row,cm[row,destination_input]]=vm[row,cm[row,input_number]]
                vm[row,cm[row,input_number]]=0
            elif (input_number==total_inputs-1)
                output[row]=vm[row,cm[row,input_number]]
        return(output)
    '''
            

            #FOLLOWIN CODE PROGRESSED EVERY SINGLE CONNECTION.ITS UNNECESARRY WHEN WE ARE JUST GOING TO CONSIDER ONE OUTPUT IN FINAL WHEN ALL INPUTS ARE PROCESSED.
##            value=0
##            for col in range (0,CLASSIFICATION_SUPPORT.no_of_connections_per_Neuron): # NO OF CONNECTIONS
##                vm2[row,cm[row,col]]=value
##                value=vm[row,cm[row,col]]
##            output[row]=value
##        return(output,vm2)




    
    def Progressing_Inputs(cm,vm):#NEW EFFICIENT SHIFTING OF connections !!!
        vm2=vm*0#  TEMPORARY VALUE MATRIX
        output=numpy.zeros((CLASSIFICATION_SUPPORT.sdr_output_columns))# FINAL OUTPUT
        for row in range (0,CLASSIFICATION_SUPPORT.sdr_output_columns): #NUMBER OF NEURONS
            if (input_number<total_inputs-1):
                vm[row,cm[row,destination_input]]=vm[row,cm[row,input_number]]
                vm[row,cm[row,input_number]]=0
            elif (input_number==total_inputs-1):
                output[row]=vm[row,cm[row,input_number]]
        return(output)

##FOLLOWIN CODE PROGRESSED EVERY SINGLE CONNECTION.ITS UNNECESARRY WHEN WE ARE JUST GOING TO CONSIDER ONE OUTPUT IN FINAL WHEN ALL INPUTS ARE PROCESSED.   
##            value=0
##            for col in range (0,CLASSIFICATION_SUPPORT.no_of_connections_per_Neuron): # NO OF CONNECTIONS
##                vm2[row,cm[row,col]]=value
##                value=vm[row,cm[row,col]]
##            output[row]=value
##        return(output,vm2)

    """
    def Progressing_Inputs(cm,vm):#shifts inputs based on connections !!! TAKING LONG TIME !!!
        vm2=vm*0#  TEMPORARY VALUE MATRIX
        output=numpy.zeros((CLASSIFICATION_SUPPORT.sdr_output_columns))# FINAL OUTPUT
        
        for row in range (0,CLASSIFICATION_SUPPORT.sdr_output_columns):
            for col in range (0,CLASSIFICATION_SUPPORT.sdr_output_columns):
                connection_value=cm[row,col]
                if connection_value == CLASSIFICATION_SUPPORT.no_of_connections_per_Neuron:
                    output[row]=vm[row,col]
                else:
                    for col2 in range (0,CLASSIFICATION_SUPPORT.sdr_output_columns):
                        if cm[row,col2]== connection_value+1:
                            vm2[row,col2] = vm[row,col]
        
        return(output,vm2)
    """

    

    
    def Binary2Sdr(binMat,cm,on_bits_sdr): #CONVERTS ANY GIVEN BINARY INPUT TO SDR OUTPUT OF SPECIFIED DIMENSIONS.UNIVERSAL!!THIS FUNCTION DOESNT DEPEND ON PARAMETERS IN THIS CLASS.

        #THIS IS FOR ONE VECTOR OF BINARY REPRESENATAION
        #binMatLength=len(binMat)
        [cm_row,cm_col]=cm.shape
        sdr=numpy.zeros((cm_row))
        #for row in range(0,binMat_row):
        sdr=sum((cm*(binMat)).T) #multiplication is weird in python 3.3
        sdr=SENTENCE.top_bit(sdr,on_bits_sdr)
        return (sdr)
        
        #THIS ONE IS FOR A MATRIX OF BINARY REPRESENTATIONS
        '''[binMat_row,binMat_col]=binMat.shape
        [cm_row,cm_col]=cm.shape
        sdr=numpy.zeros((binMat_row,cm_row))
        for row in range(0,binMat_row):
            sdr[row,:]=sum((cm*(binMat[row,:])).T) #multiplication is weird in python 3.3
            sdr[row,:]=SENTENCE.top_bit(sdr[row,:],on_bits_sdr)
        return (sdr)
        '''



    def KP_Read_1Img_MeanMat(fin,cm1_value,cm1_length,on_bits_sdr,KP_numOfOnes_in_values,KP_numOfOnes_in_group_length,KP_max_group_size,KP_distinct_values):
        
        #fin=open(r'C:\Users\Aseem\Documents\MATLAB\RESEARCH\forpy.txt','r')

        #l1=1 #SYMBOLYSIS THE THE LOOP HAS JUST STARTED
        #while(True):
            #if l1~=1: #MEANS L1 HAS READ SOME LINE AND ITS NO LONGER ==1. CONVERT THE BINARY TO SDR, LEARN IT AND STORE IT.
                
                
            
        l1=fin.readline() #READING ONE ROW AT A TIME
        l2=list(map(float,l1.split(',',7)))#BECAUSE WE HAVE 7 PARAMETERS HERE and SEPERATED BY ','. THIS GIVES US A VECTOR. INDEX 4 IS THE QUADRANT.--X, Y, VALUE, LENGTH,QUADRANT,DIST_FRM_CENTER,SUBFOLDER NUMBER
        
        value1=[] #DISTANCE FROM CENTER IN 1ST QUADRANT
        value2=[] #DISTANCE FROM CENTER IN 2ND QUADRANT
        value3=[]
        value4=[]
        length1=[] # LENGTH OF GROUPS IN 1ST QUADRANT 
        length2=[] # LENGTH OF GROUPS IN 2nd QUADRANT
        length3=[]
        length4=[]
        #print(' values and lengths initiated' )

        if l2[4]>0:#QUADRANT EXIST SO IMAGE EXISTS. TAKE ITS CLASS NUMBER
            classification_number=l2[6] # CHECK IF ITS L2[6] !!!
            
            
        while(l2[4]==1): #THIS LOOP GIVES A FINAL REPRESENTATION OF 1ST QUADRANT
            #print('while loop quadrant 1')
            value1_bin=(CLASSIFICATION_SUPPORT.Converting_input_2binary(KP_distinct_values,KP_numOfOnes_in_values,l2[2],0)) #VALUE / \ - | IN BINARY REPRESENTATION
            value1.append(CLASSIFICATION_SUPPORT.Binary2Sdr(value1_bin,cm1_value,on_bits_sdr))
             
            length1_bin=(CLASSIFICATION_SUPPORT.Converting_input_2binary(KP_max_group_size,KP_numOfOnes_in_group_length,l2[3],1)) #LENGTH OF A GROUP IN BINARY
            length1.append(CLASSIFICATION_SUPPORT.Binary2Sdr(length1_bin,cm1_length,on_bits_sdr))
            
            l1=fin.readline() #READING ONE ROW AT A TIME
            l2=list(map(float,l1.split(',',7)))#SPLITTING IT IN A VECTOR
            if l2[4] ==0: #QUADRANT ENDED
                l1=fin.readline() #READING ONE ROW AT A TIME
                l2=list(map(float,l1.split(',',7)))#SPLITTING IT IN A VECTOR
                break
                

    
        if l2[4]>0:#QUADRANT EXIST SO IMAGE EXISTS. TAKE ITS CLASS NUMBER
            classification_number=l2[6] # CHECK IF ITS L2[6] !!!
            
        while(l2[4]==2): #THIS LOOP GIVES A FINAL REPRESENTATION OF 1ST QUADRANT
            #print('while loop quadrant 2')
            value2_bin=(CLASSIFICATION_SUPPORT.Converting_input_2binary(KP_distinct_values,KP_numOfOnes_in_values,l2[2],0)) #VALUE / \ - | IN BINARY REPRESENTATION
            value2.append(CLASSIFICATION_SUPPORT.Binary2Sdr(value2_bin,cm1_value,on_bits_sdr))
             
            length2_bin=(CLASSIFICATION_SUPPORT.Converting_input_2binary(KP_max_group_size,KP_numOfOnes_in_group_length,l2[3],1)) #LENGTH OF A GROUP IN BINARY
            length2.append(CLASSIFICATION_SUPPORT.Binary2Sdr(length2_bin,cm1_length,on_bits_sdr))
            
            l1=fin.readline() #READING ONE ROW AT A TIME
            l2=list(map(float,l1.split(',',7)))#SPLITTING IT IN A VECTOR
            if l2[4] ==0: #QUADRANT ENDED
                l1=fin.readline() #READING ONE ROW AT A TIME
                l2=list(map(float,l1.split(',',7)))#SPLITTING IT IN A VECTOR
                break



        while(l2[4]==3): #THIS LOOP GIVES A FINAL REPRESENTATION OF 1ST QUADRANT
            #print('while loop quadrant 3')
            value3_bin=(CLASSIFICATION_SUPPORT.Converting_input_2binary(KP_distinct_values,KP_numOfOnes_in_values,l2[2],0)) #VALUE / \ - | IN BINARY REPRESENTATION
            value3.append(CLASSIFICATION_SUPPORT.Binary2Sdr(value3_bin,cm1_value,on_bits_sdr))
             
            length3_bin=(CLASSIFICATION_SUPPORT.Converting_input_2binary(KP_max_group_size,KP_numOfOnes_in_group_length,l2[3],1)) #LENGTH OF A GROUP IN BINARY
            length3.append(CLASSIFICATION_SUPPORT.Binary2Sdr(length3_bin,cm1_length,on_bits_sdr))
            
            l1=fin.readline() #READING ONE ROW AT A TIME
            l2=list(map(float,l1.split(',',7)))#SPLITTING IT IN A VECTOR
            if l2[4] ==0: #QUADRANT ENDED
                l1=fin.readline() #READING ONE ROW AT A TIME
                l2=list(map(float,l1.split(',',7)))#SPLITTING IT IN A VECTOR
                break



        while(l2[4]==4): #THIS LOOP GIVES A FINAL REPRESENTATION OF 1ST QUADRANT
            #print('while loop quadrant 4')
            value4_bin=(CLASSIFICATION_SUPPORT.Converting_input_2binary(KP_distinct_values,KP_numOfOnes_in_values,l2[2],0)) #VALUE / \ - | IN BINARY REPRESENTATION
            value4.append(CLASSIFICATION_SUPPORT.Binary2Sdr(value4_bin,cm1_value,on_bits_sdr))
             
            length4_bin=(CLASSIFICATION_SUPPORT.Converting_input_2binary(KP_max_group_size,KP_numOfOnes_in_group_length,l2[3],1)) #LENGTH OF A GROUP IN BINARY
            length4.append(CLASSIFICATION_SUPPORT.Binary2Sdr(length4_bin,cm1_length,on_bits_sdr))
            
            l1=fin.readline() #READING ONE ROW AT A TIME
            l2=list(map(float,l1.split(',',7)))#SPLITTING IT IN A VECTOR
            if l2[4] ==0: #QUADRANT ENDED
                l1=fin.readline() #READING ONE ROW AT A TIME
                l2=list(map(float,l1.split(',',7)))#SPLITTING IT IN A VECTOR
                break



            
        
        return (value1,value2,value3,value4,length1,length2,length3,length4,classification_number) #YAY

    def KP_Read_1Img_MeanMat_TESTING(fin,cm1_value,cm1_length,on_bits_sdr,KP_numOfOnes_in_values,KP_numOfOnes_in_group_length,KP_max_group_size,KP_distinct_values):
        
        #fin=open(r'C:\Users\Aseem\Documents\MATLAB\RESEARCH\forpy.txt','r')

        #l1=1 #SYMBOLYSIS THE THE LOOP HAS JUST STARTED
        #while(True):
            #if l1~=1: #MEANS L1 HAS READ SOME LINE AND ITS NO LONGER ==1. CONVERT THE BINARY TO SDR, LEARN IT AND STORE IT.
                
                
            
        l1=fin.readline() #READING ONE ROW AT A TIME
        l2=list(map(float,l1.split(',',7)))#BECAUSE WE HAVE 7 PARAMETERS HERE and SEPERATED BY ','. THIS GIVES US A VECTOR. INDEX 4 IS THE QUADRANT.--X, Y, VALUE, LENGTH,QUADRANT,DIST_FRM_CENTER,SUBFOLDER NUMBER
        
        value1=[] #DISTANCE FROM CENTER IN 1ST QUADRANT
        value2=[] #DISTANCE FROM CENTER IN 2ND QUADRANT
        value3=[]
        value4=[]
        length1=[] # LENGTH OF GROUPS IN 1ST QUADRANT 
        length2=[] # LENGTH OF GROUPS IN 2nd QUADRANT
        length3=[]
        length4=[]
        #print(' values and lengths initiated' )

        
            
            
        while(l2[4]==1): #THIS LOOP GIVES A FINAL REPRESENTATION OF 1ST QUADRANT
            #print('while loop quadrant 1')
            value1_bin=(CLASSIFICATION_SUPPORT.Converting_input_2binary(KP_distinct_values,KP_numOfOnes_in_values,l2[2],0)) #VALUE / \ - | IN BINARY REPRESENTATION
            value1.append(CLASSIFICATION_SUPPORT.Binary2Sdr(value1_bin,cm1_value,on_bits_sdr))
             
            length1_bin=(CLASSIFICATION_SUPPORT.Converting_input_2binary(KP_max_group_size,KP_numOfOnes_in_group_length,l2[3],1)) #LENGTH OF A GROUP IN BINARY
            length1.append(CLASSIFICATION_SUPPORT.Binary2Sdr(length1_bin,cm1_length,on_bits_sdr))
            
            l1=fin.readline() #READING ONE ROW AT A TIME
            l2=list(map(float,l1.split(',',7)))#SPLITTING IT IN A VECTOR
            if l2[4] ==0: #QUADRANT ENDED
                l1=fin.readline() #READING ONE ROW AT A TIME
                l2=list(map(float,l1.split(',',7)))#SPLITTING IT IN A VECTOR
                break
                

    
        
            
        while(l2[4]==2): #THIS LOOP GIVES A FINAL REPRESENTATION OF 1ST QUADRANT
            #print('while loop quadrant 2')
            value2_bin=(CLASSIFICATION_SUPPORT.Converting_input_2binary(KP_distinct_values,KP_numOfOnes_in_values,l2[2],0)) #VALUE / \ - | IN BINARY REPRESENTATION
            value2.append(CLASSIFICATION_SUPPORT.Binary2Sdr(value2_bin,cm1_value,on_bits_sdr))
             
            length2_bin=(CLASSIFICATION_SUPPORT.Converting_input_2binary(KP_max_group_size,KP_numOfOnes_in_group_length,l2[3],1)) #LENGTH OF A GROUP IN BINARY
            length2.append(CLASSIFICATION_SUPPORT.Binary2Sdr(length2_bin,cm1_length,on_bits_sdr))
            
            l1=fin.readline() #READING ONE ROW AT A TIME
            l2=list(map(float,l1.split(',',7)))#SPLITTING IT IN A VECTOR
            if l2[4] ==0: #QUADRANT ENDED
                l1=fin.readline() #READING ONE ROW AT A TIME
                l2=list(map(float,l1.split(',',7)))#SPLITTING IT IN A VECTOR
                break



        while(l2[4]==3): #THIS LOOP GIVES A FINAL REPRESENTATION OF 1ST QUADRANT
            #print('while loop quadrant 3')
            value3_bin=(CLASSIFICATION_SUPPORT.Converting_input_2binary(KP_distinct_values,KP_numOfOnes_in_values,l2[2],0)) #VALUE / \ - | IN BINARY REPRESENTATION
            value3.append(CLASSIFICATION_SUPPORT.Binary2Sdr(value3_bin,cm1_value,on_bits_sdr))
             
            length3_bin=(CLASSIFICATION_SUPPORT.Converting_input_2binary(KP_max_group_size,KP_numOfOnes_in_group_length,l2[3],1)) #LENGTH OF A GROUP IN BINARY
            length3.append(CLASSIFICATION_SUPPORT.Binary2Sdr(length3_bin,cm1_length,on_bits_sdr))
            
            l1=fin.readline() #READING ONE ROW AT A TIME
            l2=list(map(float,l1.split(',',7)))#SPLITTING IT IN A VECTOR
            if l2[4] ==0: #QUADRANT ENDED
                l1=fin.readline() #READING ONE ROW AT A TIME
                l2=list(map(float,l1.split(',',7)))#SPLITTING IT IN A VECTOR
                break



        while(l2[4]==4): #THIS LOOP GIVES A FINAL REPRESENTATION OF 1ST QUADRANT
            #print('while loop quadrant 4')
            value4_bin=(CLASSIFICATION_SUPPORT.Converting_input_2binary(KP_distinct_values,KP_numOfOnes_in_values,l2[2],0)) #VALUE / \ - | IN BINARY REPRESENTATION
            value4.append(CLASSIFICATION_SUPPORT.Binary2Sdr(value4_bin,cm1_value,on_bits_sdr))
             
            length4_bin=(CLASSIFICATION_SUPPORT.Converting_input_2binary(KP_max_group_size,KP_numOfOnes_in_group_length,l2[3],1)) #LENGTH OF A GROUP IN BINARY
            length4.append(CLASSIFICATION_SUPPORT.Binary2Sdr(length4_bin,cm1_length,on_bits_sdr))
            
            l1=fin.readline() #READING ONE ROW AT A TIME
            l2=list(map(float,l1.split(',',7)))#SPLITTING IT IN A VECTOR
            if l2[4] ==0: #QUADRANT ENDED
                l1=fin.readline() #READING ONE ROW AT A TIME
                l2=list(map(float,l1.split(',',7)))#SPLITTING IT IN A VECTOR
                break

 
        return (value1,value2,value3,value4,length1,length2,length3,length4) #YAY


    

    def Converting_input_2binary(n,m,k,exc_overlap): # CONVERTING INPUT (1,2,3,...) TO A BINARY REPRESENTATION WHICH IS FURTHER CONVERTED TO SDR
        
        # n- TOTAL NUMBER OF DISTINCT INPUTS IN THE PROGRAM
        # m-NUMBER OF 1'S IN EVERY BINARY REPRESENTATION NEEDED
        # k- WHICH INPUT (1ST,2ND,3RD..) NO ZEROS PLEASE !!!!
        #exc_overlap- IF 0 THEN MUTUALLY EXCLUSIVE OUTPUT NEEDED. IF 1 THEN OVERLAPING OUTPUT NEEDED
        if exc_overlap ==0:
            output=numpy.zeros((m*n)) #MUTUALY EXCLUSIVE NEEDS BIGGER SIZE.
        elif exc_overlap==1:
            output=numpy.zeros((m+n)) #MUTUALY EXCLUSIVE NEEDS BIGGER SIZE.
            
        if exc_overlap ==0:
            for i in range (1,m+1):
                output[(k*m)-i]=1
        elif exc_overlap ==1: #OVERLAPPING. WE NEED OVERLAPPING BITS FOR ANALOG SIGNALS WHERE TWO CONSECUTING INPUTS NEED OVERLAPING BITS TO PRESERVE SOME SEMANTIC SIMILARITY. BUT AS THE DIFFERENCE BETWEEN INPUTS INCREASE THE SIMILARITY SHOULD REDUCE AND EVENTUALLY VANISH
            #for i in range (0,m):
            if (k-1+(m-1))>=(m+n): # NOW IF THE INPUT IS BIGGER THAN OUR REPRESENTATION CAN REPRESENT, IT WILL JUST GIVE THE MAX AMOUNT WITHOUT CAUSING ERRORS.
                output[m+n-m:m+n]=1
            else:
                output[k-1:k-1+m]=1
        return output

    
    
    def Input_to_Sdr(self,connection_matrix_1,in_strng):#TAKES in_strng READ INPUT
        #my_str = input("Enter the sequence please \n ")
        my_str=in_strng
        sentence1=SENTENCE()  # object of class
        sent1=sentence1.RemovePunctuations(my_str) # also makes lower case
        ascii_matrix=numpy.zeros((len(sent1),CLASSIFICATION_SUPPORT.ascii_matrix_columns))
        for j in range(0,len(sent1)):
            ascii_matrix[j,:]=sentence1.Converting_char_2binary(sent1[j]) # HERE [SENTENCE.ascii_matrix_columns] IS USED FOR NO OF COLUMNS IN ASCII.
                                                                            #IT CANT BE CHANGED BY CHANING CLASSIFICATION_SUPPORT.ascii_matrix_columns
          
        self.sdr_matrix_1 = numpy.zeros((len(sent1),CLASSIFICATION_SUPPORT.sdr_output_columns))
        for row in range(0,len(sent1)):
            self.sdr_matrix_1[row,:]=sum((connection_matrix_1*(ascii_matrix[row,:])).T) #multiplication is weird in python 3.3
            self.sdr_matrix_1[row,:]=SENTENCE.top_bit(self.sdr_matrix_1[row,:],CLASSIFICATION_SUPPORT.on_bits_sdr)



    def Connection_Mat_2():#THIS NEW CM2 STORES INDEXES OF CONNECTIONS. iNDEX OF CM2 ROW (0,1,2,3..) = NEURON CONNECTIONS NUMBER (1ST CONN, 2ND CONNEC,3RD,4TH..)
                            # VALUE OF CM2[8,1,11,5] = ACTUAL VALUE MAT INDEX.
                            #CM[[2,0,0,0,4,0,0,1,0,0,3],[4,0,0,3,0,2,0,0,0,0,1]] --  CM2[[8,1,11,5],[11,6,4,1] ..] THESE IS REAL EXAMPLE 
                            #refer notebook for more !!
        cm2= numpy.zeros((CLASSIFICATION_SUPPORT.sdr_output_columns,CLASSIFICATION_SUPPORT.no_of_connections_per_Neuron))
        for row in range(0,CLASSIFICATION_SUPPORT.sdr_output_columns):
            cm2[row,:]=random.sample(range(CLASSIFICATION_SUPPORT.sdr_output_columns), CLASSIFICATION_SUPPORT.no_of_connections_per_Neuron)#random indexes [99, 36, 19, 82, 61, 10, 69, 33, 98, 37]
        return(cm2)
    
    '''
    def Connection_Mat_2(): # THIS CM IS TAKING TOO MUCH TIME FOR PROPAGATING INPUTS
        
        cm2= numpy.zeros((CLASSIFICATION_SUPPORT.sdr_output_columns,CLASSIFICATION_SUPPORT.sdr_output_columns))
        for row in range(0,CLASSIFICATION_SUPPORT.sdr_output_columns):
            indx=random.sample(range(CLASSIFICATION_SUPPORT.sdr_output_columns), CLASSIFICATION_SUPPORT.no_of_connections_per_Neuron) #random indexes [99, 36, 19, 82, 61, 10, 69, 33, 98, 37]
            random_connections=random.sample(range(1,CLASSIFICATION_SUPPORT.no_of_connections_per_Neuron+1), CLASSIFICATION_SUPPORT.no_of_connections_per_Neuron)#[4, 1, 5, 7, 10, 0, 3, 8, 9, 2]
                                                            #it will actually give 1-cnec/neurn random numbers list
            for col in range (0,CLASSIFICATION_SUPPORT.no_of_connections_per_Neuron):
                cm2[row,indx[col]]=random_connections[col]
        
        return(cm2)
    '''

    def Value_Mat(self,connection_matrix_2): #connections are added and progressed further
        output=numpy.zeros((CLASSIFICATION_SUPPORT.sdr_output_columns))# FINAL OUTPUT
        vm= numpy.zeros((CLASSIFICATION_SUPPORT.sdr_output_columns,CLASSIFICATION_SUPPORT.sdr_output_columns))
        
        for row in range(0,self.sdr_matrix_1.shape[0]): #OUTPUT CHANGES FOR EVERY INPUT.WE WILL ONLY CONSIDER THE FINAL ONE FOR CLASSIFICATION
            vm=vm+self.sdr_matrix_1[row,:] #input added
            #print('PROGRESSING INPUTS')
            (output,vm)=CLASSIFICATION_SUPPORT.Progressing_Inputs(connection_matrix_2,vm)
            #print('PROGRESSING INPUTS DONE!!')
        self.output=SENTENCE.top_bit(output,CLASSIFICATION_SUPPORT.on_bits_sdr) 
        

        



        
    def Read_txt():#READS "no_of_lines_toRead" FROM TEXT FILE AND CONVERTS IT INTO AN INPUT WORD('ajcnt')MATRIX AND CLASS NUMBER(1,2,3,4) VECTOR
        
        output_str_matrix=[]
        output_classification_vec=numpy.zeros((CLASSIFICATION_SUPPORT.no_of_lines_toRead))# CLASS WHERE INPUT BELONGS
        fin=open(CLASSIFICATION_SUPPORT.filename,'r')
        for i in range (0,CLASSIFICATION_SUPPORT.no_of_lines_toRead):
            Output_str=''
            l1=fin.readline()
            l2=l1.split(',',7)#BECAUSE WE HAVE 7 PARAMETERS HERE and SEPERATED BY ','
            
            if l2[0]=='vhigh':
                Output_str=Output_str+'a'
            elif l2[0]=='high':
                Output_str=Output_str+'b'
            elif l2[0]=='med':
                Output_str=Output_str+'c'
            elif l2[0]=='low':
                Output_str=Output_str+'d'

            if l2[1]=='vhigh':
                Output_str=Output_str+'e'
            elif l2[1]=='high':
                Output_str=Output_str+'f'
            elif l2[1]=='med':
                Output_str=Output_str+'g'
            elif l2[1]=='low':
                Output_str=Output_str+'h'

            if l2[2]=='2':
                Output_str=Output_str+'i'
            elif l2[2]=='3':
                Output_str=Output_str+'j'
            elif l2[2]=='4':
                Output_str=Output_str+'k'
            elif l2[2]=='5more':
                Output_str=Output_str+'l'

            if l2[3]=='2':
                Output_str=Output_str+'m'
            elif l2[3]=='4':
                Output_str=Output_str+'n'
            elif l2[3]=='more':
                Output_str=Output_str+'o'

            if l2[4]=='small':
                Output_str=Output_str+'p'
            elif l2[4]=='med':
                Output_str=Output_str+'q'
            elif l2[4]=='big':
                Output_str=Output_str+'r'

            if l2[5]=='low':
                Output_str=Output_str+'s'
            elif l2[5]=='med':
                Output_str=Output_str+'t'
            elif l2[5]=='high':
                Output_str=Output_str+'u'

            if l2[6]=='unacc\n':
                output_classification_vec[i]=1
            elif l2[6]=='acc\n':
                output_classification_vec[i]=2
                
            elif l2[6]=='good\n':
                output_classification_vec[i]=3
            elif l2[6]=='vgood\n':
                output_classification_vec[i]=4

            

            output_str_matrix.append(Output_str)
        return(output_str_matrix,output_classification_vec)

    def Extract_Indexes_ofOnes (self): #GIVES INDEXES OF ALL ONES FOR ONE PARTICULAR VECTOR
        indx=numpy.zeros((CLASSIFICATION_SUPPORT.on_bits_sdr))
        count=0
        for i in range (0,CLASSIFICATION_SUPPORT.sdr_output_columns):
            if self.output[i]==1:
                indx[count]=i
                count=count+1
        return(indx)

    def KP_Extract_Indexes_ofOnes (on_bits_sdr,sdr_output_columns,sdr): #GIVES INDEXES OF ALL ONES FOR ONE PARTICULAR sdr VECTOR
        indx=numpy.zeros((on_bits_sdr))
        count=0
        for i in range (0,sdr_output_columns):
            if sdr[i]==1:
                indx[count]=i
                count=count+1
        return(indx)

    def KP_Classifying(l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18,l19,l20,l21,l22,l23,l24,l25,l26,l27,l28,l29,l30,l31,l32,l33,l34,l35,l36,l37,l38,l39,l40,l41,l42,l43,l44,l45,l46,l47,l48,l49,l50,l51,l52,l53,l54,l55,l56,l57,l58,l59,l60,l61,l62,l63,l64,l65,l66,l67,l68,l69,l70,l71,l72,l73,l74,l75,l76,l77,l78,l79,l80,l81,l82,l83,l84,l85,l86,l87,l88,l89,l90,l91,l92,l93,l94,l95,l96,l97,l98,l99,l100,l101,l102,l103,l104,l105,l106,l107,l108,l109,l110,l111,l112,l113,l114,l115,l116,l117,
l118,l119,l120,l121,l122,l123,l124,classification_vec,index):
        
        if classification_vec==1:
            l1.append(index)
            
        elif classification_vec==2:
            l2.append(index)
            
        elif classification_vec==3:
            l3.append(index)
            
        elif classification_vec==4:
            l4.append(index)

        elif classification_vec==5:
            l5.append(index)
            
        elif classification_vec==6:
            l6.append(index)
            
        elif classification_vec==7:
            l7.append(index)
        elif classification_vec==8:
            l8.append(index)
            
        elif classification_vec==9:
            l9.append(index)
            
        elif classification_vec==10:
            l10.append(index)

        elif classification_vec==11:
            l11.append(index)
            
        elif classification_vec==12:
            l12.append(index)
            
        elif classification_vec==13:
            l13.append(index)
        elif classification_vec==14:
            l14.append(index)
            
        elif classification_vec==15:
            l15.append(index)
            
        elif classification_vec==16:
            l16.append(index)

        elif classification_vec==17:
            l17.append(index)
            
        elif classification_vec==18:
            l18.append(index)
            
        elif classification_vec==19:
            l19.append(index)
        elif classification_vec==20:
            l20.append(index)
            
        elif classification_vec==21:
            l21.append(index)
            
        elif classification_vec==22:
            l22.append(index)

        elif classification_vec==23:
            l23.append(index)
            
        elif classification_vec==24:
            l24.append(index)
            
        elif classification_vec==25:
            l25.append(index)
        elif classification_vec==26:
            l26.append(index)
            
        elif classification_vec==27:
            l27.append(index)
            
        elif classification_vec==28:
            l28.append(index)

        elif classification_vec==29:
            l29.append(index)
        
        elif classification_vec==30:
            l30.append(index)
            
        elif classification_vec==31:
            l31.append(index)
        elif classification_vec==32:
            l32.append(index)
            
        elif classification_vec==33:
            l33.append(index)
            
        elif classification_vec==34:
            l34.append(index)

        elif classification_vec==35:
            l35.append(index)
            
        elif classification_vec==36:
            l36.append(index)
            
        elif classification_vec==37:
            l37.append(index)
        elif classification_vec==38:
            l38.append(index)
            
        elif classification_vec==39:
            l39.append(index)
            
        elif classification_vec==40:
            l40.append(index)

        elif classification_vec==41:
            l41.append(index)
            
        elif classification_vec==42:
            l42.append(index)
            
        elif classification_vec==43:
            l43.append(index)
        elif classification_vec==44:
            l44.append(index)
        elif classification_vec==45:
            l45.append(index)
        elif classification_vec==46:
            l46.append(index)
        elif classification_vec==47:
            l47.append(index)
        elif classification_vec==48:
            l48.append(index)
        elif classification_vec==49:
            l49.append(index)
        elif classification_vec==50:
            l50.append(index)
        elif classification_vec==51:
            l51.append(index)
        elif classification_vec==52:
            l52.append(index)
        elif classification_vec==53:
            l53.append(index)
        elif classification_vec==54:
            l54.append(index)
        elif classification_vec==55:
            l55.append(index)
        elif classification_vec==56:
            l56.append(index)
        elif classification_vec==57:
            l57.append(index)
        elif classification_vec==58:
            l58.append(index)
        elif classification_vec==59:
            l59.append(index)
        elif classification_vec==60:
            l60.append(index)
        elif classification_vec==61:
            l61.append(index)
        elif classification_vec==62:
            l62.append(index)
        elif classification_vec==63:
            l63.append(index)
        elif classification_vec==64:
            l64.append(index)
        elif classification_vec==65:
            l65.append(index)
        elif classification_vec==66:
            l66.append(index)
        elif classification_vec==67:
            l67.append(index)
        elif classification_vec==68:
            l68.append(index)
        elif classification_vec==69:
            l69.append(index)
        elif classification_vec==70:
            l70.append(index)
        elif classification_vec==71:
            l71.append(index)
        elif classification_vec==72:
            l72.append(index)
        elif classification_vec==73:
            l73.append(index)
        elif classification_vec==74:
            l74.append(index)
        elif classification_vec==75:
            l75.append(index)
        elif classification_vec==76:
            l76.append(index)
        elif classification_vec==77:
            l77.append(index)
        elif classification_vec==78:
            l78.append(index)
        elif classification_vec==79:
            l79.append(index)
        elif classification_vec==80:
            l80.append(index)
        elif classification_vec==81:
            l81.append(index)
        elif classification_vec==82:
            l82.append(index)
        elif classification_vec==83:
            l83.append(index)
        elif classification_vec==84:
            l84.append(index)
        elif classification_vec==85:
            l85.append(index)
        elif classification_vec==86:
            l86.append(index)
        elif classification_vec==87:
            l87.append(index)
        elif classification_vec==88:
            l88.append(index)
        elif classification_vec==89:
            l89.append(index)
        elif classification_vec==90:
            l90.append(index)
        elif classification_vec==91:
            l91.append(index)
        elif classification_vec==92:
            l92.append(index)
        elif classification_vec==93:
            l93.append(index)
        elif classification_vec==94:
            l94.append(index)
        elif classification_vec==95:
            l95.append(index)
        elif classification_vec==96:
            l96.append(index)
        elif classification_vec==97:
            l97.append(index)
        elif classification_vec==98:
            l98.append(index)
        elif classification_vec==99:
            l99.append(index)
        elif classification_vec==100:
            l100.append(index)
        elif classification_vec==101:
            l101.append(index)
        elif classification_vec==102:
            l102.append(index)
        elif classification_vec==103:
            l103.append(index)
        elif classification_vec==104:
            l104.append(index)
        elif classification_vec==105:
            l105.append(index)
        elif classification_vec==106:
            l106.append(index)
        elif classification_vec==107:
            l107.append(index)
        elif classification_vec==108:
            l108.append(index)
        elif classification_vec==109:
            l109.append(index)
        elif classification_vec==110:
            l110.append(index)
        elif classification_vec==111:
            l111.append(index)
        elif classification_vec==112:
            l112.append(index)
        elif classification_vec==113:
            l113.append(index)
        elif classification_vec==114:
            l114.append(index)
        elif classification_vec==115:
            l115.append(index)
        elif classification_vec==116:
            l116.append(index)
        elif classification_vec==117:
            l117.append(index)
        elif classification_vec==118:
            l118.append(index)
        elif classification_vec==119:
            l119.append(index)
        elif classification_vec==120:
            l120.append(index)
        elif classification_vec==121:
            l121.append(index)
        elif classification_vec==122:
            l122.append(index)
        elif classification_vec==123:
            l123.append(index)
        elif classification_vec==124:
            l124.append(index)
##        elif classification_vec==125:
##            l125.append(index)
##        elif classification_vec==126:
##            l126.append(index)
##        elif classification_vec==127:
##            l127.append(index)








        

    def Classifying(l1,l2,l3,l4,classification_vec,index):
        
        if classification_vec==1:
            l1.append(index)
            
        elif classification_vec==2:
            #print('found class 2')
            l2.append(index)
            
        elif classification_vec==3:
            l3.append(index)
            
        elif classification_vec==4:
            l4.append(index)



    def KP_Predicting_Class_trial(list_of_list,index,KP_on_bits_sdr):# PREDICTING WHAT CLASS THE INPUT BELONGS TO MORE APPROPRIATLY
         # NUMBER OF CLASSIFICATIONS = 4.MAXIMUM TRUES IN EACH CLASS. CLASS 1= INDEX 0
                                        #CLASS 2= INDEX 1.. SO ON

        
        count=0
        for list_stored in range (0,KP_on_bits_sdr): # EVERY BIT OF NEW PREDICTING  IMAGE SDR
            for list_new in range (0,KP_on_bits_sdr): # EVERY BIT OF STORED IMAGE SDR
                if index[list_stored]== list_of_list[list_new] : # CHECKS IF EVERY BIT OF NEW IMAGE IS FOUND IN STORED IMAGE. IF YES COUNT ++
                    count=count+1
                    break
                #MOSTLY ALL INDEXES ARE IN INCREASING ORDER. SO TO MAKE IT FAST MAY BE IF INDEX VALUE INCREASES JUST BREAK COZ ITS NOT GOING TO MATCH LATER TOO LOL

        
        return(count)
                            


#,l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18,l19,l20,l21,l22,l23,l24,l25,l26,l27,l28,l29,l30,l31,l32,l33,l34,l35,l36,l37,l38,l39,l40,l41,l42,l43,l44,l45,l46,l47,l48,l49,l50,l51,l52,l53,l54,l55,l56,l57,l58,l59,l60,l61,l62,l63,l64,l65,l66,l67,l68,l69,l70,l71,l72,l73,l74,l75,l76,l77,l78,l79,l80,l81,l82,l83,l84,l85,l86,l87,l88,l89,l90,l91,l92,l93,l94,l95,l96,l97,l98,l99,l100,l101,l102,l103,l104,l105,l106,l107,l108,l109,l110,l111,l112,l113,l114,l115,l116,l117,
#l118,l119,l120,l121

#array([   6.,   12.,   69.,   76.,   80.,   86.,  103.,  116.,  126., 130.,  139.,  141.,  148.,  153.])
#array([  12.,   69.,    0.,   76.,   80.,   86.,  103.,  116.,  126.,     129.,  130.,  139.,  141.,  148.,  153.])
    def KP_Predicting_Class(list_of_list,index,KP_on_bits_sdr):# PREDICTING WHAT CLASS THE INPUT BELONGS TO MORE APPROPRIATLY
        #SINCE BUILDING LIST_OF_LIST IS DIFFICULT TO TRY OUT U CAN USE KP_Predicting_Class_trial. JUST GIVE TWO NORMAL LISTS AND IT WILL GIVE U MATCHING COUNT
        #list_of_list and index WILL HAVE INDEXES OF ONES. ALL THOSE INDEXES WILL ALWAYS BE DISTINCT, SO WHILE TESTING MAKE SURE TO USE DISTINCT INDEXES. ORDER DOESNT MATTER IN THIS AWESOME NEW FUNCTION. IT WILL STILL GIVE RIGHT MATCHES
        #eg list_of_list=[1,5,20,40]. THIS IS A PATTEREND LEARNED. index=[1,5,0,22] WILL HAVE 2 MATCHING INDICES OF ONES.index=[1,15,5,22] ALSO HAVE 2 MATCHIN INDICES BUT THE ORDER IS DIFFERENT AS OF THE LEARNT INPUT, BUT STILL THIS FUCNTION WILL GIVE OUTPUT = 2 YAY !!
        l_max_true=numpy.zeros((121)) # NUMBER OF CLASSIFICATIONS = 4.MAXIMUM TRUES IN EACH CLASS. CLASS 1= INDEX 0
                                        #CLASS 2= INDEX 1.. SO ON
        

        for lists in range(0,len(list_of_list)): #FOR L1 L2 L3 ETX..
            for i in range (0,len(list_of_list[lists])):  #FOR L1[0] L1[1] L1[2]..
                count=0
                for list_stored in range (0,KP_on_bits_sdr): # EVERY BIT OF NEW PREDICTING  IMAGE SDR
                    for list_new in range (0,KP_on_bits_sdr): # EVERY BIT OF STORED IMAGE SDR
                        if index[list_stored]== list_of_list[lists][i][list_new] : # CHECKS IF EVERY BIT OF NEW IMAGE IS FOUND IN STORED IMAGE. IF YES COUNT ++
                            count=count+1
                            break
                        #MOSTLY ALL INDEXES ARE IN INCREASING ORDER. SO TO MAKE IT FAST MAY BE IF INDEX VALUE INCREASES JUST BREAK COZ ITS NOT GOING TO MATCH LATER TOO LOL

                if l_max_true[lists]<count: # IF THIS COUNT OF TRUES IS MORE THEN PREVIOUS ONE.. STORE IT 
                    l_max_true[lists]=count
                            
                        
                    
##        #FOR THIS BELOW PATCH BOTH LISTS HAVE TO BE IN EXACT ORDER IN ORDER TO BE TRUE. BUT IN REALITY WE NEED TO CHECK IF ANY ON PIXELS OVERLAP IRRESPECTIVE OF THE ORDER OF THEIR INDICES.
            # [1 3 5]   [3 5 0] SHD HAVE 2 TRUE , BT BELOW CODE WILL GIVE ZERO TRUE.  ABOVE PATCH WILL GIVE 2 TRUE :) 
##        for lists in range(0,len(list_of_list)):
##            for i in range (0,len(list_of_list[lists])):
##                k=list_of_list[lists][i]==index
##                count=0
##                for j in range (0,len(index)): #COUNTING NUMBER OF TRUE'SS
##                    if k[j]==True:
##                        count=count+1
##                if l_max_true[lists]<count: # IF THIS COUNT OF TRUES IS MORE THEN PREVIOUS ONE.. STORE IT 
##                    l_max_true[lists]=count
        return(l_max_true)



    def KP_RemovingDuplicateLists(): #REMOVES DUPLICATE LISTS FROM LIST_OF_LISTS EVERY CLASS
        list_of_list=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Marine plankton\train\saved\list_of_list.npy')
                    #C:\Users\ahegshetye\Downloads\clt\kag\saved
                    #C:\Users\Aseem\Documents\KAGGLE\Marine plankton\train\saved
        list_of_list2=[] #NEW UNDUPLICATE LIST
        duplicates=0
        distincts=0
        for lists in range(0,len(list_of_list)): #FOR L1 L2 L3 ETX..
            l_tmp=[]
            l_tmp.append(list_of_list[lists][0]) #EVERY LISTS (L1 L2 L3..) 1ST ELEMENT IS STORED IN LIST2
            for i in range (1,len(list_of_list[lists])):  #FOR L1[0] L1[1] L1[2]..
                #IF NOT IT l_tmp THEN APPEND IT
                len_l_tmp=len(l_tmp)
                for l2 in range(0,len_l_tmp):
                    k=list_of_list[lists][i]==l_tmp[l2]
                    if k.all()==True:
                        duplicates=duplicates+1
                        print('duplicate found',duplicates)
                        print(list_of_list[lists][i])
                        print(l_tmp[l2])
                        break
                    elif l2==len_l_tmp-1: #SINCE THE LOOP DINT BREAK AND WE LOOKED ALL OVER AND MATCH IS NOT FOUND SO APPPEND IT COZ ITS NEW
                        l_tmp.append(list_of_list[lists][i])
                        distincts=distincts+1
                        

            list_of_list2.append(l_tmp)
        print(distincts)    
        return (list_of_list2)
        #numpy.save(r'C:\Users\ahegshetye\Downloads\clt\kag\saved\list_of_list_distinct',list_of_list2)
        
        
    


    def Predicting_Class(l1,l2,l3,l4,index):# PREDICTING WHAT CLASS THE INPUT BELONGS TO MORE APPROPRIATLY
        l_max_true=numpy.zeros((4)) # NUMBER OF CLASSIFICATIONS = 4.MAXIMUM TRUES IN EACH CLASS. CLASS 1= INDEX 0
                                        #CLASS 2= INDEX 1.. SO ON
        
        for i in range (0,len(l1)):
            k=l1[i]==index
            
            count=0
            for j in range (0,len(index)): #COUNTING NUMBER OF TRUE'SS
                if k[j]==True:
                    count=count+1
            if l_max_true[0]<count: # IF THIS COUNT OF TRUES IS MORE THEN PREVIOUS ONE.. STORE IT 
                l_max_true[0]=count
                k2=k
                l1_final=l1[i]
        #print ('final TRUE for l1\n', k2)
        #print ('l1_final',l1_final)
                


        for i in range (0,len(l2)):
            k=l2[i]==index
            
            
            count=0
            for j in range (0,len(index)): #COUNTING NUMBER OF TRUE'SS
                if k[j]==True:
                    count=count+1
            #print('L2 count = ',count)
            #print('L2 l_max_true[1] = ',l_max_true[1])
            if l_max_true[1]<count: # IF THIS COUNT OF TRUES IS MORE THEN PREVIOUS ONE.. STORE IT. SOMETIMES NOT EVEN A SINGLE ON BIT MATCHES. SO THAT TIME WE NEED TO MAKE SURE L2_FINAL IS ATLEAST INITIATED.SO WE TAKE <= INSTEAD OF <
                #print('L2 count = ',count)
                #print('L2 l_max_true[1] = ',l_max_true[1])
                l_max_true[1]=count
                k2=k
                l2_final=l2[i]
        #print ('final TRUE for l2\n', k2)
        #print ('l2_final',l2_final)
                

        for i in range (0,len(l3)):
            k=l3[i]==index
            count=0
            for j in range (0,len(index)): #COUNTING NUMBER OF TRUE'SS
                if k[j]==True:
                    count=count+1
            if l_max_true[2]<count: # IF THIS COUNT OF TRUES IS MORE THEN PREVIOUS ONE.. STORE IT 
                l_max_true[2]=count


        for i in range (0,len(l4)):
            k=l4[i]==index
            count=0
            for j in range (0,len(index)): #COUNTING NUMBER OF TRUE'SS
                if k[j]==True:
                    count=count+1
            if l_max_true[3]<count: # IF THIS COUNT OF TRUES IS MORE THEN PREVIOUS ONE.. STORE IT 
                l_max_true[3]=count

        return(l_max_true)

    def CM33(cm33,sdr3_1,sdr3_2): #LEARNING CONNECTIONS !!
        #print(sdr3_2.shape[0],'\n its a 1D vector preety sure\n')
            
        for i in range(0,sdr3_2.shape[0]):
            if sdr3_2[i]==1:
                cm33[i,:]=cm33[i,:]+(sdr3_1)

        cm33=SENTENCE2.thresholding_of_matrix_2(cm33,1) #thresholding_of_matrix_2(cm33,SENTENCE2.threshold)-- THRESHOLD  HERE IS 1 BECAUSE WE WANT IT TO LEARN PATTERNS THE MOMENT IT SEES ONE.nO NEED OF REPEATED LEARNING!
            
        return(cm33)




    def thresholding_of_matrix_2(matrx,thresh):
        #all above threshold = threshold .. othrs are left untouched.. !!NOT MADE 0
        mat=matrx*1
        if len(matrx.shape)==3:
            for i in range(0,matrx.shape[0]):
                for j in range(0,matrx.shape[1]):
                    for k in range(0,matrx.shape[2]):
                        if (matrx[i,j,k])>thresh:
                            mat[i,j,k]=thresh
        elif len(matrx.shape)==2:
            for i in range(0,matrx.shape[0]):
                for j in range(0,matrx.shape[1]):
                    if (matrx[i,j])>thresh:
                        mat[i,j]=thresh

        elif len(matrx.shape)==1:
            for i in range(0,matrx.shape[0]):
                if (matrx[i])>thresh:
                        mat[i]=thresh
                        

        return(mat)
