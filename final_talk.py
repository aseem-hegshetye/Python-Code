import sys
import operator
import math
import time
import numpy
#from numpy import *



from TALK_support import *
from TALK_support2 import *
def main():
    fresh_start=-1 #it shows whether u have freshly started all matrix [1] or loaded previous ones[0]
    inp=input('hi.. Do you want to load the existing memory[0] or start fresh[1].\nIf starting fresh please dont save it, coz it will override existing memory and not enhance it.\n')
    if inp =='0': #load existing memory
        fresh_start=0
        connection_matrix_1=numpy.load(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\connection_matrix_1.npy')
        cm1_down=numpy.load(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\cm1_down.npy')

        synapse_matrix_1=numpy.load(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\synapse_matrix_1.npy')
        synapse_matrix_1_talk1=numpy.load(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\synapse_matrix_1_talk1.npy')
        connection_matrix_2=numpy.load(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\connection_matrix_2.npy')
        cm2_down=numpy.load(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\cm2_down.npy')
        synapse_matrix_2=numpy.load(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\synapse_matrix_2.npy')
        synapse_matrix_2_talk1=numpy.load(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\synapse_matrix_2_talk1.npy')
        connection_matrix_3=numpy.load(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\connection_matrix_3.npy')
        cm3_down=numpy.load(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\cm3_down.npy')
        inhibition_mat2_1strow=numpy.load(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\inhibition_mat2_1strow.npy')
        inhibition_mat2=numpy.load(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\inhibition_mat2.npy')
        inhibition_mat1_1strow=numpy.load(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\inhibition_mat1_1strow.npy')
        inhibition_mat1=numpy.load(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\inhibition_mat1.npy')
        cm33=numpy.load(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\cm33.npy')
        print('all matrix loaded from memory \n')
        
    elif inp=='1': #initialize fresh matrices
        fresh_start=1
        #all dimensions here are defined by sentence2
        connection_matrix_1=numpy.random.rand(SENTENCE2.memory_matrix_columns,SENTENCE2.ascii_matrix_columns)
        cm1_down=numpy.zeros((SENTENCE2.ascii_matrix_columns,SENTENCE2.memory_matrix_columns))
        
        #connection_matrix_1=SENTENCE2.Connection_Matrix_1(connection_matrix_1) #makes 50% NaN
        synapse_matrix_1=numpy.random.rand(SENTENCE2.memory_matrix_rows*SENTENCE2.memory_matrix_columns,SENTENCE2.memory_matrix_rows,SENTENCE2.memory_matrix_columns)
        synapse_matrix_1_talk1=synapse_matrix_1*1
        connection_matrix_2=numpy.random.rand(SENTENCE2.memory_matrix_columns,SENTENCE2.memory_matrix_rows,SENTENCE2.memory_matrix_columns)
        #connection_matrix_2=SENTENCE2.Connection_Matrix_2(connection_matrix_2) #makes 50% NaN
        cm2_down=numpy.zeros((SENTENCE2.memory_matrix_columns,SENTENCE2.memory_matrix_rows,SENTENCE2.memory_matrix_columns))#its for predicting mm
        synapse_matrix_2=numpy.random.rand(SENTENCE2.memory_matrix_rows*SENTENCE2.memory_matrix_columns,SENTENCE2.memory_matrix_rows,SENTENCE2.memory_matrix_columns)
        synapse_matrix_2_talk1=synapse_matrix_2*1
        connection_matrix_3=numpy.random.rand(SENTENCE2.memory_matrix_columns,SENTENCE2.memory_matrix_rows,SENTENCE2.memory_matrix_columns)
        cm3_down=numpy.zeros((SENTENCE2.memory_matrix_columns,SENTENCE2.memory_matrix_rows,SENTENCE2.memory_matrix_columns))#its for predicting mm2
        #connection_matrix_3=SENTENCE2.Connection_Matrix_2(connection_matrix_3) #function CM2 is similar to CM3
        
        #INHIBITION MATRIX ARE SPECIFIC TO A MEMORY MATRIX
        inhibition_mat2_1strow=numpy.ones((SENTENCE2.memory_matrix_columns,SENTENCE2.memory_matrix_columns))*-1
        inhibition_mat2=numpy.ones(((SENTENCE2.memory_matrix_rows-1)*SENTENCE2.memory_matrix_columns,SENTENCE2.memory_matrix_rows-1,SENTENCE2.memory_matrix_columns))*-1
        inhibition_mat1_1strow=numpy.ones((SENTENCE2.memory_matrix_columns,SENTENCE2.memory_matrix_columns))*-1
        inhibition_mat1=numpy.ones(((SENTENCE2.memory_matrix_rows-1)*SENTENCE2.memory_matrix_columns,SENTENCE2.memory_matrix_rows-1,SENTENCE2.memory_matrix_columns))*-1
        #ZEROFYING ALL INHIBITION MATRICES..ZEROING OWN DIMENSIONS !!
        inhibition_mat2_1strow=SENTENCE2.inhibition_mat_zeroing_own_dimension(inhibition_mat2_1strow)
        inhibition_mat2=SENTENCE2.inhibition_mat_zeroing_own_dimension(inhibition_mat2)
        inhibition_mat1_1strow=SENTENCE2.inhibition_mat_zeroing_own_dimension(inhibition_mat1_1strow)
        inhibition_mat1=SENTENCE2.inhibition_mat_zeroing_own_dimension(inhibition_mat1)
        cm33=numpy.zeros((SENTENCE2.memory_matrix_columns,SENTENCE.memory_matrix_columns))
        print('all matrixes initiated FRESHLY\n')

    while True:
        numpy.set_printoptions(threshold=numpy.nan)
        numpy.set_printoptions(linewidth=999999)
        
        my_str = input("Enter first sentence: ")
        if not my_str:
            break
        sentence1=SENTENCE()  # object of class 
        sent1=sentence1.RemovePunctuations(my_str) # also makes lower case
        sent2=sentence1.SeparateWords(sent1) #sent2= ['how','are','you']
        #print (sentence1.number_of_words)
        sentence1.NumberofWords(sent2)
        #print (sentence1.number_of_words)
        sentence1.Length_of_each_word(sent2)
        
        sentence1.Ascii_Matrix()  # self.ascii_matrix has ascii matrix now !!
        
        
        connection_matrix_1=sentence1.Sdr_Matrix_1(connection_matrix_1) #store self.sdr_matrix_1
        
        
        synapse_matrix_1_talk1=sentence1.Memory_Matrix_1(synapse_matrix_1_talk1) # builts self.memory_matrix_1
        
        connection_matrix_2=sentence1.Sdr_Matrix_2(connection_matrix_2) #store self.sdr_matrix_2
        
        
        synapse_matrix_2_talk1=sentence1.Memory_Matrix_2(synapse_matrix_2_talk1) # builts self.memory_matrix_2
        
        
        connection_matrix_3=sentence1.Sdr_Matrix_3(connection_matrix_3) #store self.sdr_matrix_3
        
        
        connection_matrix_3_added=sentence1.Connection_matrix_sum_of_OnSdr(sentence1.sdr_matrix_3,connection_matrix_3)
        
        
       

        #----------------------------------------------------------------------------------------
        #  TALK 2 STARTS
        my_str2 = input("Enter a response please: ")
        sentence2=SENTENCE2()  # object of class 
        
        if not my_str2: # EMPTY STRING
            print('let me predict the response')
            #we need to predict the sentence2.sdr3 and also generate output from it
            sentence2.sdr_matrix_3=(sentence1.sdr_matrix_3*cm33).sum(axis=1) # one ascii row
            sentence2.sdr_matrix_3=SENTENCE2.top_bit(sentence2.sdr_matrix_3,SENTENCE2.on_bits_sdr3)
        else:
            
            sent1=sentence2.RemovePunctuations(my_str2) # also makes lower case
            sent2=sentence2.SeparateWords(sent1) #sent2= ['how','are','you']
            
            sentence2.NumberofWords(sent2)
            
            sentence2.Length_of_each_word(sent2)
           
            sentence2.Ascii_Matrix()  # self.ascii_matrix has ascii matrix now !!
            
            
            connection_matrix_1=sentence2.Sdr_Matrix_1(connection_matrix_1,cm1_down) #store self.sdr_matrix_1...strengthens cm1_down from ascii mat
            #print('sdr matrix 1 set')
            
            
            (synapse_matrix_1,inhibition_mat1)=sentence2.Memory_Matrix_1(synapse_matrix_1,inhibition_mat1) # builts self.memory_matrix_1
            #print('memory mat 1 set')
            
            
           
            connection_matrix_2=sentence2.Sdr_Matrix_2(connection_matrix_2,cm2_down) #store self.sdr_matrix_2.. strengthens cm2_down from mm1
            #print('sdr mat 2 set')
            
            
            (synapse_matrix_2,inhibition_mat2)=sentence2.Memory_Matrix_2(synapse_matrix_2,inhibition_mat2) # builts self.memory_matrix_2
            #print('memory mat 2 set')
            
            #[connection_matrix_3,connection_matrix_3_MOTOR]=sentence2.Sdr_Matrix_3(connection_matrix_3,connection_matrix_3_MOTOR) #store self.sdr_matrix_3
            connection_matrix_3=sentence2.Sdr_Matrix_3(connection_matrix_3,cm3_down) #store self.sdr_matrix_3.. strengthens cm3_down from mm2
            
            cm33=SENTENCE2.CM33(cm33,sentence1.sdr_matrix_3,sentence2.sdr_matrix_3) #STRENGHTNENING CM33 ONLY IF USER GIVES A RESPONS.
            #WE DONT STRENTHEN IT IF ALGORITHM HAS PREDICTED THE RESPONSE, COZ ALGORITHM MAY BE WRONG AND WE DONT WANT IT TO LEARN WRONG THINGS :)

            inhibition_mat2_1strow=SENTENCE2.INHIBITION_MATRIX2_1strow(inhibition_mat2_1strow,sentence2.sdr_matrix_2[0])
            #inhibition mat has connections 1 or -1

            inhibition_mat1_1strow=sentence2.INHIBITION_MATRIX1_1strow(inhibition_mat1_1strow)
        #-----------------------------------------------------------
        #BELOW SDR3
        
        
        mm2_pred=sentence2.PREDICTING_MM2(cm3_down)
        
        #if sum(sum(numpy.logical_xor(mm2_pred,sentence2.memory_matrix_2)))==0:
            #input(' prediction exactly matches.. \nyahoo !!! \n' )
        #SENTENCE2.Predicted_or_not(mm2_pred,sentence2.memory_matrix_2)

        #INHIBITION STARTS
        
        
        
        
        inhibited_mm2_1strow=sentence2.INHIBITED_MM2_1strow(mm2_pred,inhibition_mat2_1strow)
        #inhibited_mm has the memory matrix after being inhibited with inhibition_mat. <0 -- = 0 .... >=0 -- =1
        
        inhibited_mm2=sentence2.INHIBITED_MM2(mm2_pred,inhibition_mat2,inhibited_mm2_1strow,synapse_matrix_2)
        
        try:
            if (sentence2.sdr2_down == sentence2.sdr_matrix_2).all() == True:
                #print(' YAHOOOO!!! u r great sir !  SDR 2 predicted Exactly \n')
                print('')
            else:
                #print((sentence2.sdr2_down == sentence2.sdr_matrix_2).astype(int))
                #print(' above 0 is mismatch\n')
                print(' prediction NOT matched\n' )
        except:
            #print(' some error occured in SDR2 comparison  :D lol\n' )
            print('')

        #-------------------------------------------------------------------
        
        # Below SDR2
        
        if sentence2.sdr2_down.all()!=1:
            #print('BELOW SDR2 loop ON.')
            mm1_pred=sentence2.PREDICTING_MM(cm2_down) # mm_pred is 3d matrix. one matrix for one word
            
            #print(' PREDICTING_MM done\n')

            
            
            #print('INHIBITION_MATRIX1_1strow  done\n')

            inhibited_mm1_1strow=sentence2.INHIBITED_MM1_1strow(mm1_pred,inhibition_mat1_1strow)
            
            #print('INHIBITED_MM1_1strow  done\n')

            sentence2.INHIBITED_MM1(mm1_pred,inhibition_mat1,inhibited_mm1_1strow,synapse_matrix_1)
            
            #print(' INHIBITED_MM1 done\n' )
            

            try:
                if (sentence2.sdr1_down == sentence2.sdr_matrix_1).all() == True:
                    #print(' YAHOOOO!!! u r great sir ! SDR 1 predicted Exactly \n')
                    print('')
                else:
                    #print((sentence2.sdr1_down == sentence2.sdr_matrix_1).astype(int))
                    #print(' above 0 is mismatch\n')
                    print(' prediction NOT matched\n' )
            except:
                #print(' some error occured in SDR1 comparison :D lol\n' )
                print('')
            #print('BELOW SDR2 loop ended\n')
            #self.sdr1_down is a 3D matrix !!! i have cheacked. it will atleast have[0,0,:].. no to many index error
            sentence2.ascii_mat_down=SENTENCE2.ASCII_matrix_from_SDR1(sentence2.sdr1_down,cm1_down)
            output=sentence2.CHARACTERS_OUTPUT()
            #print(output,'\n Response reproduced\n')
            print('\n--',''.join(output),'\n\n')
        

        #print(sentence1.sdr_matrix_3)
        #print('\n aobe is SDR matrix 3 sentence1 \n ')
        #print(sentence2.sdr_matrix_3)
        #print('\n aobe is SDR matrix 3 sentence2 response\n ')

    if fresh_start==1:
        print('\n all matrix were freshly started. if u save these, previous data will be overwritten, unless this is ur first time and no matrix existed before\n')
    elif fresh_start==0:
        print('\n all matrix were loaded from memory. u can choose to SAVE them \n')
        
    save=input('do u want to save new connections[1] or not [0]? \n')
    if save=='1': # SAVING ALL MATRICES
        numpy.save(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\connection_matrix_1',connection_matrix_1)
        numpy.save(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\cm1_down',cm1_down)
        numpy.save(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\synapse_matrix_1',synapse_matrix_1)
        numpy.save(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\synapse_matrix_1_talk1',synapse_matrix_1_talk1)
        numpy.save(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\connection_matrix_2',connection_matrix_2)
        numpy.save(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\cm2_down',cm2_down)
        numpy.save(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\synapse_matrix_2',synapse_matrix_2)
        numpy.save(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\synapse_matrix_2_talk1',synapse_matrix_2_talk1)
        numpy.save(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\connection_matrix_3',connection_matrix_3)
        numpy.save(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\cm3_down',cm3_down)
        numpy.save(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\inhibition_mat2_1strow',inhibition_mat2_1strow)
        numpy.save(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\inhibition_mat2',inhibition_mat2)
        numpy.save(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\inhibition_mat1_1strow',inhibition_mat1_1strow)
        numpy.save(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\inhibition_mat1',inhibition_mat1)
        numpy.save(r'C:\Users\Aseem\Documents\GROK\CODE\CLA TALK\final_talk_matrices\cm33',cm33)
        print('all matrix SAVED IN the memory \n')
    elif save=='0':
        print(' \n nothing saved.. !! ')
        
if __name__ == '__main__':
  main()
