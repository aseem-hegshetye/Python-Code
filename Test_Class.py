
#PRINTING ANYTHING ON THE SCREEN TAKES LONG TIME
#CHANGED YEARS 4 ON BITS PER SDR TO ACTUAL ON_BITS_SDR
    
import sys
import operator
import math
import time
import numpy
import csv
import copy
#from numpy import *

from TALK_support import *
from TALK_support2 import *
from CLASSIFICATION_SUPPORT import *
from Test_Class import *

##
##def main():
##    on_bits_sdr=8
####    sdr_output_columns=400
####    [all_sdrs,classValues]=Test_Class.Kaggle_Restaurant_sdrs(sdr_output_columns,on_bits_sdr) # classValues = ALL VALUES OF CLASSES FOR A PARTICULAR CASE. CASE [0] HAS CLASS VALUE classValues[0]
####    indexOfOnes=Test_Class.Kaggle_Restaurant_IndexOfOnes(all_sdrs,on_bits_sdr,sdr_output_columns)
####    [cm2,list_of_list]=Test_Class.Kaggle_Restaurant_Building_CM2(indexOfOnes,on_bits_sdr,classValues)
##    
##    
##    
##if __name__ == '__main__':
##  main()


class Test_Class():

        # Test_Class.Train()
    def Train():
        #HERE WE DONT USE WEIGHTS FOR SDR. COZ IN HERE WE ONLY PICK INDEXES. WE DONT MULTIPLY SDRS WITH CM2 TO GET OUTPUT HERE.  AND ALL DISTINCT INDEXE PATTERNS FOR A CASE ARE STORED IN CM2 AS NEURONS.
        #RATHER USING WEIGHT WILL DISTORT OUR ALGORITHM COZ [INDEX OF ONES] FUNCTION ONLY PICKS INDEX OF 1'S AND NOT ALL THOSE ARE >0
        on_bits_sdr=8
        sdr_output_columns=400
        x=[]

        #READING WHOLE EXCEL FILE FIRST
        f=open(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\train.csv','rt') #C:\Users\ahegshetye\Downloads\clt\kag\restaurnt\train2.csv        #C:\Users\Aseem\Documents\KAGGLE\Restaurant\train2.csv
        reader=csv.reader(f)
        for row in reader:
            x.append(row)

        #LOOP FOR EVERY CASE FROM EXCEL FILE
        all_sdrs=[]
        all_classes=[]
        for excel_rows in range(1,len(x)): #0TH INDEX HAS TITLES
            [sdr_PerCase,classValues]=Test_Class.Kaggle_Restaurant_sdrs(sdr_output_columns,on_bits_sdr,x,excel_rows) # classValues = ALL VALUES OF CLASSES FOR A PARTICULAR CASE. CASE [0] HAS CLASS VALUE classValues[0]
            all_sdrs.append(sdr_PerCase)
            all_classes.append(classValues[0]) #TO AVOID MAKING LIST OF LIST [[]] WHICH CREATES PROBLEMS FURTHER IN TEST Test_Class.Kaggle_Restaurant_Building_CM2//CLASSIFICATION_SUPPORT.KR_Classifying


        indexOfOnes=Test_Class.Kaggle_Restaurant_IndexOfOnes(all_sdrs,on_bits_sdr,sdr_output_columns,2)#test_train = 2
##        return(all_classes,all_sdrs,indexOfOnes)
        [cm2,list_of_list]=Test_Class.Kaggle_Restaurant_Building_CM2(indexOfOnes,on_bits_sdr,all_classes)
        numpy.save(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\cm2',cm2)
        numpy.save(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\list_of_list',list_of_list)
        print('cm2 and list_of_list SAVED')
    
##        return(cm2,list_of_list)


    ## o=Test_Class.testing()
    def testing():
        on_bits_sdr=8
        sdr_output_columns=400
        cm2=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\cm2.npy')
        list_of_list=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\list_of_list.npy')
        x=[]
        cm2_kholaleli=Test_Class.cm2_khola(cm2,sdr_output_columns)
        list_of_list_kholaleli=Test_Class.class_khola(list_of_list,len(cm2),on_bits_sdr)
        output=[]
        #DOING 1 CASE AT A TIME. THIS MAKES COMPUTER FAST AND KEEPS RAM EMPTY

        #READING WHOLE EXCEL FILE FIRST
        f=open(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\test.csv','rt') #C:\Users\ahegshetye\Downloads\clt\kag\restaurnt\train2.csv        #C:\Users\Aseem\Documents\KAGGLE\Restaurant\train2.csv
        reader=csv.reader(f)
        for row in reader:
            x.append(row)

        #LOOP FOR EVERY CASE FROM EXCEL FILE
        for excel_rows in range(1,len(x)): #0TH INDEX HAS TITLES
            
            #BUILDING ALL SDRS PER CASE #CHANGE THE EXCEL READER TO POINT TO TEST2 FILE.
            #IF INPUT FOR A PATTERN IN SDR IS ZERO IN TEST ( FOR MB TYPE OR A CITY NAME THAT DOESNT EXIST IN TRAIN) THEN SDR FOR THAT PATTERN BECOMES 0 AND EVERYTHING WORKS PERFECT. THAT PATTERN DOESNT DISTURB THE FINAL OUTPUT AND ITS AWESOME.
            [all_sdrs,classValues]=Test_Class.Kaggle_Restaurant_sdrs(sdr_output_columns,on_bits_sdr,x,excel_rows) # classValues = CLASS FOR CURRENT CASE
            
    ##        indexOfOnesSdr=Test_Class.Kaggle_Restaurant_IndexOfOnes(all_sdrs,on_bits_sdr,sdr_output_columns,1)#test_train = 2
    



            #ADD WEIGHTS TO SDR REPRESENTATION .. TESTED AND WORKING
            all_sdrs=Test_Class.KR_WeightedSDR(all_sdrs,len(cm2[0])) #(sdr,no_of_connections_per_Neuron)
            

            
            final_output=Test_Class.final_output(all_sdrs,cm2_kholaleli,on_bits_sdr) # TOP on_bits_sdr ARE 1 OTHERS ARE 0.final_output = NEURONS THAT REPRESENT THE INPT PATTERN MOST APPROPRIATLY.ITS OUTPUT FOR ALL CASES.
    ##        numpy.save(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\final_output_test',final_output)
##            print(' final_output saved ')
            
    
            
    ##        numpy.save(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\list_of_list_kholaleli_test',list_of_list_kholaleli)
    

##            print('starting predicting_class_fromKholaleli')
            list_of_list_Magnitude=Test_Class.predicting_class_fromKholaleli(final_output[0],list_of_list_kholaleli) #final_output HAS TO BE 1'S 0'S. CANT BE INDEXES
            
            #SELECT  MAGNITUDE OF TOP CLASS FOR EVERY CASE NOW
            list_of_list_BestClass_PerCase=Test_Class.selectingBestClass(list_of_list_Magnitude) #GIVES BEST CLASS NUMBER THAT MATCHES THE INPUT SEQUENCE. WE HAVE ADDED 1 SINCE INDEXING IN PYTHON STARTS FROM 0

            output.append(list_of_list_BestClass_PerCase[0])
##            break

            #TO PRING OUTPUT IN THE END SO WE CAN COPY IN EXCEL.
##        for i in range(0,len(o)):
##	    print(o[i][0])
            
##        for i in range(0,len(output)):
##            print(output[i])
        return(output)

     #  Test_Class.KR_WeightedSDR
    def KR_WeightedSDR(sdr,no_of_connections_per_Neuron): #SETS WEIGHT OF INPUTS AND THEN MULTIPLIES SDR AND GIVES APPROPRIATE REPRESENTATION. HIGH WEIGHT INPUT PRODUCES HIGH VALUE SDR SO CHANGE IN THAT INPUT NOTICIEBLY AFFECTS THE OUTPUT
        #THIS FUNCTION WONT WORK FOR KAGGLE RESTAURANT FUNCTION COZ I HAVE DECLARED WEIGHTS INSIDE THIS FUNCTION NOW. IT WILL WORK FOR TEST_CLASS RESTAURANT
        #no_of_connections_per_Neuron= len(cm2[0]) -- THAT IS THE NUMBER OF PARAMETERS PER CASE
        #DECLARING WEIGHTS INSIDE THIS FUNCTION ITSELF.
        
        
        weight=numpy.ones((no_of_connections_per_Neuron)) #IF ANY PARTICULAR INPUT IS IMPORTANT WE INCREASE ITS WEIGHTAGE. SO CHANGE IN THAT INPUT DEFINETLY AFFECTS THE OUTPUT. WE SET THEN INITIALLY TO 1 SO ORIGINAL SDR VALUE REMAINS INTACT.FURTHER WE CHANGE SOME VALUES IN THE WEIGHTAGE FUNCTION BELOW
        weight[0]=4 #MONTH_YEAR
####        weight[1]=0 #CITY NAME
        weight[2]=2 #CITY GROUP -- [Big Cities, Others]
        weight[3]=3 # TYPE OF RESTAURANT
        
        #MAKING THINGS FAST
##        for i in range (0,len(sdr)):
##            sdr[i]=sdr[i]* weight[i]
        sdr[0]=sdr[0]* weight[0]
        sdr[2]=sdr[2]* weight[2]
        sdr[3]=sdr[3]* weight[3]

        
        return(sdr)
    


        # Test_Class.selectingBestClass
    def selectingBestClass(list_of_list_Magnitude): #TAKES THE HIGHEST/BEST MATCHING CLASS FOR EVERY CASE.
        list_of_list_BestClass=[]
##        for case in range(0,len(list_of_list_Magnitude)):
        list_of_list_BestClass.append((list(list_of_list_Magnitude).index(max(list_of_list_Magnitude)))+1) # INDEXING IN PYTHON STARTS FROM 0 SO WE ADD 1 TO GET CORRECT CLASS NUMBER
        return(list_of_list_BestClass)



        ## Test_Class.predicting_class_fromKholaleli
    def predicting_class_fromKholaleli(op_kholalela,list_of_list_kholaleli): #PREDICTING WHAT CLASS THE FINAL KHOLALELA REPRESENTATION BELONGS TOO FROM A KHOLALELI CM2.. HAHAHAH
        #TESTING  WORKING FINE !! :) 
##        op_kholalela=[[1,1,1,0,0,0,0,0,0,0],[0,1,1,1,0,0,0,0,0,0]]
##        list_of_list_kholaleli=[[[1,1,1,0,0,0,0,0,0,0],[0,1,1,0,0,0,0,0,1,0]],[[1,1,1,0,0,0,0,0,0,0]],[[0,0,0,1,1,1,0,0,0,0],[1,1,1,0,0,0,0,0,0,0]]]

##        list_of_list_Magnitude=[] # IT HAS MAGNITUDE OF ALL CLASES FOR ALL CASES. FOR EG: 1 CASE WILL HAVE 139 VALUES THAT INDICATES ITS OVERLAP WITH EACH CLASS
##        for cases in range (0,len(op_kholalela)):
        
        list_of_list_Magnitude_perCase=[] #[4 10 21 5 6 9] MAGNITUDE OF OVERLAP WITH A PARTICULAR CLASS.4 IS THE MAGNITUDE OF OVERLAP WITH CLASS 1,10 WITH CLASS 2 ETX
        for classes in range(0,len(list_of_list_kholaleli)):
            class_Magnitude=[] # IT WILL HAVE ALL REPRESENTATIONS MAGNITUDES FROM A CLASS.
            for rep in range(0,len(list_of_list_kholaleli[classes])): #EVERY CLASS MAY HAVE DIFFERENT NUMBER OF REPRESENTATIONS
                new_rep_magnitude=sum([a*b for a,b in zip(list_of_list_kholaleli[classes][rep], op_kholalela)]) # MULTIPLICATION OF LIST IN PYTHON IS WEIRD, UNLIKE ARRAYS
                if not class_Magnitude: #IF THIS IS THE FIRST REPRESENTATION OR THE ONLY REPRESENTATION OF THIS CLASS
                    class_Magnitude.append(new_rep_magnitude)
                elif class_Magnitude[0] < new_rep_magnitude: # IF THIS NEW REP MATCHES THE OUPUT MORE THEN STORE THE NEW MAGNITUDE IN THE CLASS MAGNITUDE INDEX
                    class_Magnitude[0]=new_rep_magnitude
            list_of_list_Magnitude_perCase.append(class_Magnitude)
##        list_of_list_Magnitude.append(list_of_list_Magnitude_perCase)
        return(list_of_list_Magnitude_perCase)
                                      
                    
                    

        
        # Test_Class.final_output
    def final_output(input_sdr,cm2_kholaleli,on_bits_sdr): #GIVES A FINAL REPRESENTATION FOR ALL CASE IN TEST.. ITS PROCESSING IN A BUNCH
        final_output=[]
##        for cases in range(0,len(input_sdr)):
        final_rep=numpy.zeros((len(cm2_kholaleli))) # MAGNITUDE OF FIRING OF EVERY NEURON FROM CM2 FOR THIS PARTICULAR CASE
        for neuron_in_cm2 in range (0,len(cm2_kholaleli)):
            final_rep[neuron_in_cm2]=sum(sum([a*b for a,b in zip(input_sdr, cm2_kholaleli [neuron_in_cm2])])) #sum(sum(input_sdr[cases] * cm2_kholaleli [neuron_in_cm2]))
        final_rep2=SENTENCE.top_bit(final_rep,on_bits_sdr) #CONVERTS ALL TOP BITS TO 1 AND OTHERS TO 0
        final_output.append(final_rep2) # INSTEAD OF RETURNING final_rep2 I AM APENDING IT TO A LIST AND THEN RETURNING THE LIST final_output COZ I DONT WANT TO MESS UP WITH FORMATS AGAIN.MESSING UP WITH FORMATS WILL REQUIRE ME TO MAKE SOME CHANGES IN FURTHER FUNCTIONS.
        return(final_output)


        # Test_Class.class_khola
    def class_khola(list_of_list,no_of_neurons,on_bits_sdr): #no_of_neurons = len(cm2)
        #TESTING
##        cm2=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\cm2.npy')
##        list_of_list=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\list_of_list.npy')
##        no_of_neurons = len(cm2)
##        on_bits_sdr=8
        
        list_of_list_kholaleli=[]
        for classes in range(0,len(list_of_list)):
            rep_kholalela=[] # IT WILL HAVE ALL REPRESENTATIONS FROM A CLASS. THEY ALL WILL BE KHOLALELE
            for rep in range(0,len(list_of_list[classes])): #EVERY CLASS MAY HAVE DIFFERENT NUMBER OF REPRESENTATIONS
                representation=numpy.zeros((no_of_neurons)) #no_of_neurons = TOTAL NEURONS IN CM2= TOTAL DISTINCT PATTERNS IN THE LEARNT INPUT
                for neurons in range(0,on_bits_sdr): #THERE ARE on_bits_sdr NUMBER OF BITS IN EVERY CLASS.
                    representation[list_of_list[classes][rep][neurons]]=1
                rep_kholalela.append(representation)
            list_of_list_kholaleli.append(rep_kholalela)
        return(list_of_list_kholaleli)


            # Test_Class.cm2_khola
    def cm2_khola(cm2,sdr_output_columns): #CONVERTS CM2 FROM INDEXS TO BINARY 1'S AND 0'S
        #IN CM2 EVERY ROW IS A NEURON. EVERY ROW BECOMES A MATRIX OF DIMENSION [PARAMETER (ROWS)x SDR_OP_COL (COLS)]
        cm2_kholaleli=[]
        for neurons in range(0,len(cm2)):
            cm2_kholaleli_neuron=[] #IT WILL ONLY HAVE 1 CM2 NEURON REPRESENTATION AT A TIME. WE KEEP APPENDING EVERY NEURON TO FINAL cm2_kholaleli WHICH HAS ALL NEURONS REPRENTATION
            for parameters in range(0,len(cm2[0])): #PARAMETERS ARE SAME FOR ALL NEURONS IN CM2.
                kholalela_neuron_parameter=numpy.zeros((sdr_output_columns)) #PARAMETER OF A NEURON BEING CONSIDERED
                indx=cm2[neurons][parameters]
                kholalela_neuron_parameter[indx]=1 # SETTING THAT PARTICULAR INDEX OF THAT PARAMETER FOR THAT NEURON =1
                cm2_kholaleli_neuron.append(kholalela_neuron_parameter) # 1 WHOLE CM2 NEURON IS BEING CONSTRUCTED
            cm2_kholaleli.append(cm2_kholaleli_neuron)
        return(cm2_kholaleli)
                                 
        
    def Kaggle_Restaurant_Building_CM2(indexOfOnes,on_bits_sdr,classes): #indexOfOnes= FOR ALL PARAMETERS OF ALL CASES.
        # classes = ALL VALUES OF CLASSES FOR A PARTICULAR CASE. CASE [0] HAS CLASS VALUE classes[0]
        
        #TESTING
##        indexOfOnes=[]
##        indexOfOnes.append([[2,4],[6,7]])
##        indexOfOnes.append([[2,5],[10,6]])
##        classes=[]
##        classes.append(1)
##        classes.append(2)
##        indexOfOnes.append([[1,2],[3,4]])
##        indexOfOnes.append([[1,2],[6,3]])
##        indexOfOnes.append([[5,10],[3,4]])
##        indexOfOnes.append([[2,5],[3,4]])
##        indexOfOnes.append([[7,8],[9,11]])
##        on_bits_sdr=2
        #print(' indexOfOnes after initiating \n' , indexOfOnes)
        
        cm2=[]
        outputIndex=[] #HAS INDEXES OF ONES FOR FINAL OUTPUT OF THE CURRENT CASE
        allClasses=[] # ALL CASES CLASSES
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
#NOT POSSIBLE THAT NO CASE HAS FINAL REPRESENTATION THAT HAD NEURONS THAT WERE REPEATED FROM CM2. 
        list_of_list=[]
    # 1ST CASE INDEXES WILL DIRECTLY BE STORED IN THE CM2
        print(' 1st case CM2 building' )
        indexPerCaseTranspose=list(map(list,zip(*indexOfOnes[0]))) 
        for i in range(0,on_bits_sdr):
            #print(' cm2 before appending 1st indexofOnes \n' , cm2)
            cm2.append(indexPerCaseTranspose[i])  # THIS IS WORKING PERFECT
            outputIndex.append(i) # SINCE ITS 1ST CASE. ALL INDEXES (0: onbitsdsr) BELONG TO CLASS classValues[0]
##            currentClass.append(len(cm2)-1)   !!!!!!!!!
        #NOW WE HAVE INDEXES FOR OUR FIRST CLASS VALUE. LETS APPEND IT TO THE CLASS LIST IT BELONGS TO. WE ALREADY HAVE A FUNCTION FOR THAT YAY
        CLASSIFICATION_SUPPORT.KR_Classifying(l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18,l19,l20,l21,l22,l23,l24,l25,l26,l27,l28,l29,l30,l31,l32,l33,l34,l35,l36,l37,l38,l39,l40,l41,l42,l43,l44,l45,l46,l47,l48,l49,l50,l51,l52,l53,l54,l55,l56,l57,l58,l59,l60,l61,l62,l63,l64,l65,l66,l67,l68,l69,l70,l71,l72,l73,l74,l75,l76,l77,l78,l79,l80,l81,l82,l83,l84,l85,l86,l87,l88,l89,l90,l91,l92,l93,l94,l95,l96,l97,l98,l99,l100,l101,l102,l103,l104,l105,l106,l107,l108,l109,l110,l111,l112,l113,l114,l115,l116,l117,
l118,l119,l120,l121,l122,l123,l124,l125,l126,l127,l128,l129,l130,l131,l132,l133,l134,l135,l136,l137,l138,l139,classes[0],outputIndex)
        
        

    #FOR ALL REMAINING CASES
        for case in range(1,len(indexOfOnes)):
            print('CM2 building for case ',case )
            outputIndex=[] #    EVERY CASE WE START A FRESH OUTPUT LIST
            
            [indx,outputIndex]=Test_Class.Kaggle_Restaurant_UnseenNeurons(cm2,indexOfOnes[case],on_bits_sdr,outputIndex) #RETURNS THOSE INDEXES/CONNECTION SEQUENCE OF NEURONS THAT ARE NOT ALREADY STORED IN CM2. SO WE HAVE NON DUPLICATE CONNECTIONS SEQUENCES FROM GIVEN INDEXOFONES
            print('unseen neurons done' )
            for i in range(0,len(indx)):
                cm2.append(indx[i])
                outputIndex.append(len(cm2)-1)   #APPENDING INDEX OF CM2 WHERE RECENTLY A NEW NEURON (WITH NEW UNSEEN CONNECTION PATTERN) WAS ADDED

            #ARRANGING outputIndex IN ASCENDING ORDER !!!!!!!!!!!!!!!!! DO WE HAVE INT OR FLOAT.ITS INDEX SO INT.
            outputIndex=sorted(outputIndex, key=int)

            #STORING OUTPUT INDEX IN APROPRIATE CLASS LIST BEFORE WE START NEXT CASE
            CLASSIFICATION_SUPPORT.KR_Classifying(l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18,l19,l20,l21,l22,l23,l24,l25,l26,l27,l28,l29,l30,l31,l32,l33,l34,l35,l36,l37,l38,l39,l40,l41,l42,l43,l44,l45,l46,l47,l48,l49,l50,l51,l52,l53,l54,l55,l56,l57,l58,l59,l60,l61,l62,l63,l64,l65,l66,l67,l68,l69,l70,l71,l72,l73,l74,l75,l76,l77,l78,l79,l80,l81,l82,l83,l84,l85,l86,l87,l88,l89,l90,l91,l92,l93,l94,l95,l96,l97,l98,l99,l100,l101,l102,l103,l104,l105,l106,l107,l108,l109,l110,l111,l112,l113,l114,l115,l116,l117,
l118,l119,l120,l121,l122,l123,l124,l125,l126,l127,l128,l129,l130,l131,l132,l133,l134,l135,l136,l137,l138,l139,classes[case],outputIndex)

        #NOW STORE ALL L1-L139 LISTS INTO FINAL LIST_OF_LIST.
        CLASSIFICATION_SUPPORT.KR_Converting_Lists2_ListOfList(l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18,l19,l20,l21,l22,l23,l24,l25,l26,l27,l28,l29,l30,l31,l32,l33,l34,l35,l36,l37,l38,l39,l40,l41,l42,l43,l44,l45,l46,l47,l48,l49,l50,l51,l52,l53,l54,l55,l56,l57,l58,l59,l60,l61,l62,l63,l64,l65,l66,l67,l68,l69,l70,l71,l72,l73,l74,l75,l76,l77,l78,l79,l80,l81,l82,l83,l84,l85,l86,l87,l88,l89,l90,l91,l92,l93,l94,l95,l96,l97,l98,l99,l100,l101,l102,l103,l104,l105,l106,l107,l108,l109,l110,l111,l112,l113,l114,l115,l116,l117,
l118,l119,l120,l121,l122,l123,l124,l125,l126,l127,l128,l129,l130,l131,l132,l133,l134,l135,l136,l137,l138,l139,list_of_list)
    
            
        return(cm2,list_of_list)

    def test(i):
        j=i*1
        del j[0]
        return (j)

    def Kaggle_Restaurant_UnseenNeurons(cm2,indexOfOnesPerCase,on_bits_sdr,outputIndex) : # CHECKS IF LEARNT CM2 NEURONS ALREADY EXIST IN THIS NEW INDEXOFONES MATRIX FOR THIS PARTICULAR CASE, AND RETURNS ALL NEW UNSEEN NEURONS WITH FRESH NEW CONNECTIONS WHICH NEED TO BE LEARNT AND STORED IN CM2
        #indexOfOnesPerCase SHOULD BE THE REGULAR FORMAT. NOT THE TRANSPOSED ONE. COLUMNS SHD BE NEURONS (on_bits_sdr), ROWS SHD BE PARAMETERS
        #cm2 = ROWS SHD BE NEURONS ( NUMBER OF O/P NEURONS ,CAN BE DIFFERENT FROM SDR_OP_COL) , COLS SHD BE PARAMETERS (41)
        #TESTING
##        cm2=[[4, 1], [6, 3], [2, 6]]
##        indexOfOnesPerCase=[[2,3],[1,4]]
##        on_bits_sdr=2
##        i=[[2,8,6],[7,3,4],[9,7,11],[6,10,8]]
##        indexOfOnesPerCase=list(map(list,zip(*i)))
##        cm2=[[2,3,4],[6,7,8]]
        #print('indexOfOnesPerCase in unseen 1' ,indexOfOnesPerCase )
        indx1=copy.deepcopy(indexOfOnesPerCase)
        indx2=copy.deepcopy(indexOfOnesPerCase)
        for cm2neurons in range(0,len(cm2)): #LOOPS ON ALL CM2 NEURONS
            #print(' NEURON = ',cm2neurons+1)
            seqMatched=1 # 1 = YES 0 = NO
            indx1=copy.deepcopy(indx2) # RESTARTING INDX1 SO IF PREVIOUSLY DELETED SOME INDEXES AND WHICH WERE NOT MEANT TO BE DELETED, WE REBUILD THIS MATRIX .
            for parameter in range(0,len(cm2[cm2neurons])): #CHECKS IF THAT BIT IS FOUND IN ANY OF THE INDEXOFONES NEURONs  SAME PARAMETER
                #print(' parameter = ',parameter)
                value1=cm2[cm2neurons][parameter]
                if value1 not in indx2[parameter]: # IF CM2 VALUE BIT EXIST IN INDX OFONES ROW OF NEURONS ,, THEN CHECK FOR OTHER ONES, ELSE BREAK
                    seqMatched=0
                    
                    
                    #print('value dint match..BREAKING')
                    break # THIS INDXOFONES IS NEW SEQUENCE. IT SHOULD BE STORED IN CM2 AS A NEW NEURON
                else: # MEANS CM2 VALUE EXIST IN THE INDX LIST
                    
                    #indx1 [parameter][index]   EVERY PARAMETER IS AN ARRAY AS OF NOW. MAKE SURE THAT PARAMETERS ARE LIST WHEN BEING APPENDED TO INDEXOFONES.
                    del indx1[parameter][list(indx1[parameter]) .index(value1)] #DELETING THAT INDEX FROM INDX1.
                    
##        return(indx1,indx2,parameter,cm2neurons,indexOfOnesPerCase,cm2)
##                    print('AFTER  MOTHER FUCEER DELETIONS \n indx2= ',indx2)
##                    print('indx1= ',indx1)
##                    
##                    print('indexOfOnesPerCase in unseen 3' ,indexOfOnesPerCase )
##                    print('deleted from indx1 \n', indx1)
            if seqMatched ==1: #CM2 NEURON IS REPEATED IN THE INDX MATRIX SO DELETE THOSE INDEX MATRIX INDICES COZ WE DONT NEED DUPLICATES IN CM2.SET THE DELETED INDX1 AS INDX2 
##                print(' seqMatched=1' )
##                print('indx2= ',indx2)
##                print('indx1= ',indx1)
                indx2=copy.deepcopy(indx1)
                outputIndex.append(cm2neurons) # THIS CM2NEURON EXIST IN THE NEW CASE INDEX OF ONES. WE TAKE INDEX OF THIS CM2NEURON AS IT IS NOW A PART OF THE CLASS REPRESENTING THIS CASE OF INPUT
##                print('indx2= ',indx2)
##                print('indx1= ',indx1)
            else: #SEQUENCE DID NOT MATCH. SO ITS A NEW SEQUENCE. KEEP THE INDX1 MATRIX AS IT WAS BEFORE. IF FEW BITS MATCHED THEN THOSE WOULD HAVE BEEN DELETED , WE NEED TO RESTORE THEM COZ ALL BITS DONT MATCH
##                print(' seqMatched=0' )
##                print('indx2= ',indx2)
##                print('indx1= ',indx1)
                indx1=copy.deepcopy(indx2) # THOUGH WE HAVE THIS UP JUST WHEN A NEW NEURON LOOP STARTS, IF ITS THE LAST NEURON AND CHANGES HAPPEN , FLOW WONT GO UP AGAIN. IT WILL JUST RETURN WHAT IT HAS IN THIS VARIABLE. SO WE NEED TO CHANGE IT HERE TOO.
                
##                print('indx2= ',indx2)
##                print('indx1= ',indx1)
##                print('indexOfOnesPerCase in unseen 4' ,indexOfOnesPerCase )
        return(list(map(list,zip(*indx2))),outputIndex) #RETURNING FINAL INDEX OF ONES BY  REMOVING SEQUENCES THAT ALREADY EXIST IN CM2.OUTPUT FORMAT IS JUST LIKE OUR CM2 SO JUST APPEND IT YAY HOLA.
            
                
        
    #[indexOfOnes]=Test_Class.Kaggle_Restaurant_IndexOfOnes(all_sdrs,on_bits_sdr,sdr_output_columns)
    def Kaggle_Restaurant_IndexOfOnes(sdr,on_bits_sdr,sdr_output_columns,test_train):
        parametersPerCase=len(sdr[0]) # THERE ARE 136 INPUTS IN TRAIN.. EVERY INPUT IS A SEQ OF 41 PATTERNS. LETS SAY EVERY ROW IN EXCEL IS A CASE. EVERY CASE HAS 41 PARAMETERS.
        indexOfOnes=[] #ALL CASES indexOfOnes[0][0] = 1ST CASE 1ST PARAMETER INDEX OF ONES.
        
        
        for case in range(0,len(sdr)):
            indexOfOnesPerCase=[] # ALL PARAMETERS PER CASE
            for parameter in range(0,parametersPerCase):#GIVES DISTINCT CONNECTION PATTERNS FOR THE FIRST CASE
                indexOfOnesPerCase.append(CLASSIFICATION_SUPPORT.KR_Extract_Indexes_ofOnes (on_bits_sdr,sdr_output_columns,sdr[case][parameter],test_train)) #GIVES INDEXES OF ALL ONES FOR ONE PARTICULAR sdr VECTOR [test_train- 1= test 2 = train]
            indexOfOnes.append(indexOfOnesPerCase)

        return(indexOfOnes)

    
    def Kaggle_Restaurant_sdrs(sdr_output_columns,on_bits_sdr,x,excel_rows): #GIVES BUNCH OF SDRS FOR ALL INPUT SEQUENCES
        
##        no_of_connections_per_Neuron=42 #VERY IMPORTANT. #39 +3. IT IS ONLY USED TO SET THE CM2 WHICH IS A bunch OF INDEXES OF CONNECTIONS FOR ALL NEURONS.EVEN IF ITS > THEN THE ACTUAL INPUT COUNT ,MY EXCELLENT PROGRAMING SKILLS TAKE CARE OF IT. MY HEALTH VALUE MAT PROGRESSES INPUTS BASED ON ACTUAL INPUT SDR LENGTH. AND GIVES OUTPUT WHEN ITS RIGHT
##                                        #SO JUST SET no_of_connections_per_Neuron ANYTHING ABOVE ACTUAL NUMBER OF INPUTS.
##        input_weightage=numpy.ones((no_of_connections_per_Neuron)) #IF ANY PARTICULAR INPUT IS IMPORTANT WE INCREASE ITS WEIGHTAGE. SO CHANGE IN THAT INPUT DEFINETLY AFFECTS THE OUTPUT. WE SET THEN INITIALLY TO 1 SO ORIGINAL SDR VALUE REMAINS INTACT.FURTHER WE CHANGE SOME VALUES IN THE WEIGHTAGE FUNCTION BELOW
##        
##        total_noOf_classes=139 #TOTAL CLASSES FOR CLASSIFICATION
        maxNoOfDistinctExclusivPatternsPossible=sdr_output_columns/on_bits_sdr #FOR ANY GIVEN SDR LENGTH AND % OF ON BITS
        
##        x=[]
    ###########  TRAIN

        # TO CONVERT IT TO TESTING JUST CHANGE THE OPEN FILE NAME. SEE IF BLANK CLASS NUMBER AFFECTS THE CODE
##        f=open(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\test.csv','rt') #C:\Users\ahegshetye\Downloads\clt\kag\restaurnt\train2.csv        #C:\Users\Aseem\Documents\KAGGLE\Restaurant\train2.csv
##        reader=csv.reader(f)
##        for row in reader:
##            x.append(row)

##        all_sdrs=[] #HAS ALL SDRS FOR EVERY INPUT SEQUENCE. all_sdrs[1] WILL HAVE 41 SDRS (41 INPUT SEQUENCES IN 1 INPUT)
        classValues=[] # HAS ALL CLASS NUMBERS. IF CASE 1 BELONGS TO CLASS 5 THEN classValues[1] =5...INDEX STARTS FROM 0
##        for excel_rows in range(1,len(x)): #0TH INDEX HAS TITLES
##            print('count of excel rows read ')
        print(excel_rows)
        ip=[]
        sdr_rep=[]

        #OPEN DATE MONTH_YEAR.
        #IF IT BECOMES NEGATIVE OR 0 DONT WORRY ABOUT IT. KR_Converting_input_2SDR WILL GIVE US A 0 SDR
        ip.append(math.ceil((int(x[excel_rows][2])/5)))# DIVIDE BY 5 SO THAT FOR 5 MONTHS IT HAS SAME VALUE. 1 BIT CHANGES AFTER 5 MONTHS

        

        #CITY NAME
        if x[excel_rows][4] =='#N/A': #IF CITY DOESNT EXIST PUT 0
            ip.append(0)
        else:
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
##            break


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
            ip.append((int(x[excel_rows][i]))) # NOW WE PREPROCESS THE EXCEL AND ROUND ALL P'S AFTER ADDING 1 TO ELIMINATE 0'S IN VALUES.
            



        #REVENUE
        ip.append(int(x[excel_rows][45])) #45TH COLUMN HAS THE REVENUE CLASS
        #ip.append(math.floor(int(x[excel_rows][45])/1000000) ) #45th COLUMN HAS THE REVENUE AMOUNT

##            print(ip)
##            input(' IP LOOKS GOOD? ')

##### SDR'S

        # FOR 8 ON BIT ONE SDR 400 BITS TOTAL. 50 TOTAL EXCLUSIVE REPRESENTATIONS POSSIBLE. cITY GROUP NEVER WILL BE 0.

        
        #MONTH_YEAR
        sdr_rep.append(CLASSIFICATION_SUPPORT.KR_Converting_input_2SDR(sdr_output_columns,on_bits_sdr,ip[0],1))
        

        #CITY NAME
        
        sdr_rep.append(CLASSIFICATION_SUPPORT.KR_Converting_input_2SDR(sdr_output_columns,on_bits_sdr,ip[1],0))
     
        

##      #CITY GROUP

      
        #CITY GROUP IS NEVER 0 SO ADDED IP[] WITH SOMETHING IS FINE DURING SDR CONVERSION
        sdr_rep.append(CLASSIFICATION_SUPPORT.KR_Converting_input_2SDR(sdr_output_columns,on_bits_sdr,(maxNoOfDistinctExclusivPatternsPossible-10+ip[2]),0)) #40 + INPUT  SO IT MAY BE 41 OR 42
        


##      #TYPE
        #WE ARE MULTIPLYING IP[] SO THAT IF ITS 0 THEN OUTPUT WILL REMAIN 0.. NO EXTRA NOISE WILL BE ADDED.
        sdr_rep.append(CLASSIFICATION_SUPPORT.KR_Converting_input_2SDR(sdr_output_columns,on_bits_sdr,(math.floor((maxNoOfDistinctExclusivPatternsPossible/3)*ip[3])),0)) 
        


##      #P1 - P37
        
        #25 IS MAX VALUE IN P'S IN TRAIN.. 31 IS MAX IN TEST
        for i in range(4,41): #ip[41] HAS THE CLASS VALUE

            sdr_rep.append(CLASSIFICATION_SUPPORT.KR_Converting_input_2SDR(sdr_output_columns,on_bits_sdr,((math.ceil((i/4)*(i/4)*3))+(ip[i]*2)),1))
                #MAX OF (i/4)*(i/4)*3)  WILL BE 300 .
                #(ip[i]*2) -- [*2] IS DONE SO THAT 2 BITS CHANGE WHEN THE ip[i] INCREASES BY 1. WHEN ip[i] changes BY 5, OUTPUT WILL BE COMPLETELY DIFERENT REPRESENTATION

        #HERE WE HAVE THE BATCH SDR REP FOR THIS SEQUENCE OF INPUT ( ONE INP BATCH). LETS CREATE A CM2 HERE.
##        all_sdrs.append(sdr_rep)
        classValues.append(ip[41])
        
        return(sdr_rep,classValues)

    def testing_timeOfExc():
        

        cm2=numpy.load(r'C:\Users\Aseem\Documents\KAGGLE\Restaurant\cm2.npy')
        cm2_kholaleli=numpy.zeros((41,400)) # AL BINARY
        indxOfOnes=numpy.zeros((41,400)) # AL BINARY

        #OPENING UP 1ST NEURON FROM CM2
        for parameters in range(0,len(cm2[0])):
            cm2_kholaleli[parameters] [cm2[0][parameters]] =1

        #LETS SAY CM2 2ND NEURON IS NOW INDEX OF ONES MATRIX. LETS CONVERT THAT TOO TO BINARY. KHOLUN TAKUYA.
        #OPENING UP 1ST NEURON FROM CM2
        for parameters in range(0,len(cm2[1])):
            indxOfOnes[parameters] [cm2[1][parameters]] =1

            
            
        x=[]
        start_time = time.time()
        for i in range(0,100000): #1000 NEURONS IN 1 CM2. SO THIS IS TIME TAKEN PER CASE
            #print('method 1 case = ',i)
            x.append((sum(sum(indxOfOnes*cm2_kholaleli))))
        
        print('time taken by method 1 =', (time.time() - start_time)) #0.16600990295410156
        #print('x =',x)
        y=0
        x=[]
        start_time = time.time()
        for i in range(0,100000): #1000 NEURONS IN 1 CM2. SO THIS IS TIME TAKEN PER CASE
            #print(' method 2 case = ',i)
            for parameters in range(0,len(cm2[0])):
                y=y+indxOfOnes[parameters][cm2[0][parameters]]
            x.append(y) 
                           
        print('time taken by method 2 =', (time.time() - start_time))# 0.16600894927978516
        #print('x =',x)
        #return(x)

        



