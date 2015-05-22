
import sys
import operator
import math
import time
import numpy
#from numpy import *



from TALK_support2 import *
def main():
    count=0
    while True:
        
        if count==0:
            connection_matrix_1=numpy.random.rand(SENTENCE2.memory_matrix_columns,SENTENCE2.ascii_matrix_columns)
            cm1_down=numpy.zeros((SENTENCE2.ascii_matrix_columns,SENTENCE2.memory_matrix_columns))
            #connection_matrix_1=SENTENCE2.Connection_Matrix_1(connection_matrix_1) #makes 50% NaN
            synapse_matrix_1=numpy.random.rand(SENTENCE2.memory_matrix_rows*SENTENCE2.memory_matrix_columns,SENTENCE2.memory_matrix_rows,SENTENCE2.memory_matrix_columns)
            connection_matrix_2=numpy.random.rand(SENTENCE2.memory_matrix_columns,SENTENCE2.memory_matrix_rows,SENTENCE2.memory_matrix_columns)
            #connection_matrix_2=SENTENCE2.Connection_Matrix_2(connection_matrix_2) #makes 50% NaN
            cm2_down=numpy.zeros((SENTENCE2.memory_matrix_columns,SENTENCE2.memory_matrix_rows,SENTENCE2.memory_matrix_columns))#its for predicting mm
            synapse_matrix_2=numpy.random.rand(SENTENCE2.memory_matrix_rows*SENTENCE2.memory_matrix_columns,SENTENCE2.memory_matrix_rows,SENTENCE2.memory_matrix_columns)
            connection_matrix_3=numpy.random.rand(SENTENCE2.memory_matrix_columns,SENTENCE2.memory_matrix_rows,SENTENCE2.memory_matrix_columns)
            cm3_down=numpy.zeros((SENTENCE2.memory_matrix_columns,SENTENCE2.memory_matrix_rows,SENTENCE2.memory_matrix_columns))#its for predicting mm2
            #connection_matrix_3=SENTENCE2.Connection_Matrix_2(connection_matrix_3) #function CM2 is similar to CM3
            cmm3=numpy.random.rand(SENTENCE2.memory_matrix_columns,SENTENCE2.cmm3_rows,SENTENCE2.memory_matrix_columns) #connection matrix motor 3
            cmm3=SENTENCE2.CMM3(cmm3) #builds connection matrix motor with first 50% sdr having 25% of rows/words
            #INHIBITION MATRIX ARE SPECIFIC TO A MEMORY MATRIX
            inhibition_mat2_1strow=numpy.ones((SENTENCE2.memory_matrix_columns,SENTENCE2.memory_matrix_columns))*-1
            inhibition_mat2=numpy.ones(((SENTENCE2.memory_matrix_rows-1)*SENTENCE2.memory_matrix_columns,SENTENCE2.memory_matrix_rows-1,SENTENCE2.memory_matrix_columns))*-1
            inhibition_mat1_1strow=numpy.ones((SENTENCE2.memory_matrix_columns,SENTENCE2.memory_matrix_columns))*-1
            inhibition_mat1=numpy.ones(((SENTENCE2.memory_matrix_rows-1)*SENTENCE2.memory_matrix_columns,SENTENCE2.memory_matrix_rows-1,SENTENCE2.memory_matrix_columns))*-1
            #ZEROFYING ALL INHIBITION MATRICES !!
            inhibition_mat2_1strow=SENTENCE2.inhibition_mat_zeroing_own_dimension(inhibition_mat2_1strow)
            inhibition_mat2=SENTENCE2.inhibition_mat_zeroing_own_dimension(inhibition_mat2)
            inhibition_mat1_1strow=SENTENCE2.inhibition_mat_zeroing_own_dimension(inhibition_mat1_1strow)
            inhibition_mat1=SENTENCE2.inhibition_mat_zeroing_own_dimension(inhibition_mat1)
            print('all matrixes initiated\n')

        numpy.set_printoptions(threshold=numpy.nan)
        numpy.set_printoptions(linewidth=999999)
        
        my_str = input("Enter a sentence: ")
        sentence1=SENTENCE2()  # object of class 
        sent1=sentence1.RemovePunctuations(my_str) # also makes lower case
        sent2=sentence1.SeparateWords(sent1) #sent2= ['how','are','you']
        #print (sentence1.number_of_words)
        sentence1.NumberofWords(sent2)
        #print (sentence1.number_of_words)
        sentence1.Length_of_each_word(sent2)
        #print (sentence1.length_of_each_word)
        #print(sentence1.words)
        sentence1.Ascii_Matrix()  # self.ascii_matrix has ascii matrix now !!
        
        #print (connection_matrix_1)
        connection_matrix_1=sentence1.Sdr_Matrix_1(connection_matrix_1,cm1_down) #store self.sdr_matrix_1...strengthens cm1_down from ascii mat
        print('sdr matrix 1 set')
        #input('connection_matrix1')
        #print(connection_matrix_1)
        
        (synapse_matrix_1,inhibition_mat1)=sentence1.Memory_Matrix_1(synapse_matrix_1,inhibition_mat1) # builts self.memory_matrix_1
        print('memory mat 1 set')
        #print (sentence1.memory_matrix_1)
        #input('above is  Main() memmory matrix 1')
        #print(sentence1.sdr_matrix_1)
        #input('above is self.sdr_matrix_1')
        #print (synapse_matrix_1)
        #input('above is Main() synapse_matrix_1')
        
        #print(connection_matrix_2)
        #input('\n above is connection_matrix_2 \n')
        connection_matrix_2=sentence1.Sdr_Matrix_2(connection_matrix_2,cm2_down) #store self.sdr_matrix_2.. strengthens cm2_down from mm1
        print('sdr mat 2 set')
        #print(sentence1.sdr_matrix_2)
        #print(connection_matrix_2)
        #print(sentence1.memory_matrix_1[0,:,:])
        #input(cm2_down[1:5,:,:])
        
        (synapse_matrix_2,inhibition_mat2)=sentence1.Memory_Matrix_2(synapse_matrix_2,inhibition_mat2) # builts self.memory_matrix_2
        print('memory mat 2 set')
        #print (sentence1.memory_matrix_2)
        #input('above is  memmory matrix 2')
        #print (synapse_matrix_2)
        #input(connection_matrix_3_MOTOR)
        #[connection_matrix_3,connection_matrix_3_MOTOR]=sentence1.Sdr_Matrix_3(connection_matrix_3,connection_matrix_3_MOTOR) #store self.sdr_matrix_3
        connection_matrix_3=sentence1.Sdr_Matrix_3(connection_matrix_3,cm3_down) #store self.sdr_matrix_3.. strengthens cm3_down from mm2
        #print(sentence1.sdr_matrix_1)
        #print('\nabove is sdr_matrix_1\n')
        #print(sentence1.sdr_matrix_2)
        #print('\nabove is sdr_matrix_2\n')
        
        #-----------------------------------------------------------
        #BELOW SDR3
        
        #print (cmm3[5,:,:])
        #print('above is cmm3 built. its 6th dimension dimensional\n')
        mm2_pred=sentence1.PREDICTING_MM2(cm3_down)
        #print (sentence1.memory_matrix_2)
        #print('above is  memmory matrix 2')
        #print (mm2_pred)
        #print('above is  memmory matrix 2 predicted !!')
        #if sum(sum(numpy.logical_xor(mm2_pred,sentence1.memory_matrix_2)))==0:
            #input(' prediction exactly matches.. \nyahoo !!! \n' )
        SENTENCE2.Predicted_or_not(mm2_pred,sentence1.memory_matrix_2)

        #INHIBITION STARTS
        
        #print(inhibition_mat2)
        #print('above is inhibition_mat2 \n')
        inhibition_mat2_1strow=SENTENCE2.INHIBITION_MATRIX2_1strow(inhibition_mat2_1strow,sentence1.sdr_matrix_2[0])
        #inhibition mat has connections 1 or -1
        #print(inhibition_mat_1strow)
        inhibited_mm2_1strow=sentence1.INHIBITED_MM2_1strow(mm2_pred,inhibition_mat2_1strow)
        #inhibited_mm has the memory matrix after being inhibited with inhibition_mat. <0 -- = 0 .... >=0 -- =1
        #print(inhibited_mm2_1strow)
        #print('MEMORY MATRIX 2 _1strow AFTER INHIBITION \n')
        inhibited_mm2=sentence1.INHIBITED_MM2(mm2_pred,inhibition_mat2,inhibited_mm2_1strow,synapse_matrix_2)
        #print(inhibited_mm2)
        #print('above is final inhibited memory matrix 2\n')
        #print(sentence1.sdr2_down)
        #print('above is sdr2_down\n')
        #print(sentence1.sdr_matrix_2)
        #print('above is sdr matrix 2\n')
        try:
            if (sentence1.sdr2_down == sentence1.sdr_matrix_2).all() == True:
                print(' YAHOOOO!!! u r great sir !  SDR 2 predicted Exactly \n')
            else:
                print((sentence1.sdr2_down == sentence1.sdr_matrix_2).astype(int))
                print(' above 0 is mismatch\n')
                print(' prediction NOT matched\n' )
        except:
            print(' some error occured in SDR2 comparison  :D lol\n' )

        #-------------------------------------------------------------------
        
        # Below SDR2
        
        if sentence1.sdr2_down.all()!=1:
            print('BELOW SDR2 loop ON.')
            mm1_pred=sentence1.PREDICTING_MM(cm2_down) # mm_pred is 3d matrix. one matrix for one word
            #print((mm1_pred.shape))
            #input(' shape of mm1_pred in talk2 after PREDICTING_MM\n')
            #print(mm1_pred)
            #print('above is mm_pred memory matrix predicted\n')
            print(' PREDICTING_MM done\n')

            inhibition_mat1_1strow=sentence1.INHIBITION_MATRIX1_1strow(inhibition_mat1_1strow)
            #print(inhibition_mat1_1strow)
            #print('above is inhibition_mat1_1strow \n')
            print('INHIBITION_MATRIX1_1strow  done\n')

            inhibited_mm1_1strow=sentence1.INHIBITED_MM1_1strow(mm1_pred,inhibition_mat1_1strow)
            #print(sentence1.sdr1_1strow_down)
            #print(' above is sdr matrix 1st row predicted coming down\n')
            #print(sentence1.sdr_matrix_1[:,0,:])
            #print(' above is sdr matrix 1st row going up\n')
            #print((mm1_pred.shape))
            #input(' shape of mm1_pred in talk2 after INHIBITED_MM1_1strow \n')
            print('INHIBITED_MM1_1strow  done\n')

            sentence1.INHIBITED_MM1(mm1_pred,inhibition_mat1,inhibited_mm1_1strow,synapse_matrix_1)
            
            print(' INHIBITED_MM1 done\n' )
            #print(sentence1.sdr1_down)
            #print('above is sdr1_down  predicted \n')
            #print(sentence1.sdr_matrix_1)
            #print(' above is sdr1 matrix going up\n')

            try:
                if (sentence1.sdr1_down == sentence1.sdr_matrix_1).all() == True:
                    print(' YAHOOOO!!! u r great sir ! SDR 1 predicted Exactly \n')
                else:
                    print((sentence1.sdr1_down == sentence1.sdr_matrix_1).astype(int))
                    print(' above 0 is mismatch\n')
                    print(' prediction NOT matched\n' )
            except:
                print(' some error occured in SDR1 comparison :D lol\n' )
            print('BELOW SDR2 loop ended\n')
            #self.sdr1_down is a 3D matrix !!! i have cheacked. it will atleast have[0,0,:].. no to many index error
            sentence1.ascii_mat_down=SENTENCE2.ASCII_matrix_from_SDR1(sentence1.sdr1_down,cm1_down)
            output=sentence1.CHARACTERS_OUTPUT()
            print(output)
            #print(sentence1.ascii_mat_down)
            #print('above is ascii_mat_down \n')
            #print(sentence1.ascii_matrix)
            #input('above is original ascii matrix\n')

        #MOTOR STARTS
        '''
        sentence1.MMM2() #forming Memory matrix motor 2 
        #print(sentence1.mmm2)
        #print('above is mmm2\n')
        cmm3=sentence1.SDR3_MOTOR(cmm3) #also strengthens cmm3
        #print(sentence1.sdr3_motor)
        #print(' above is SDR3 MOTOR \n' )
        #print(sentence1.sdr_matrix_3)
        #print(' above is SDR matrix 3 NON- MOTOR \n' )
        mmm2_predicted=sentence1.Predicting_MMM2(cmm3)
        # its the final added, thresholded, top bitted cmm3 for sdr2
        #print(mmm2_predicted)
        #print(' above is mmm2_predicted\n' )
        #print(sentence1.sdr_matrix_2)
        #print('above is sdr matrix 2\n')
        #print(sentence1.mmm2)
        #print(mmm2_predicted)
        if sum(sum(numpy.logical_xor(mmm2_predicted,sentence1.mmm2)))==0:
            #input(' prediction exactly matches.. \nyahoo !!! \n' )
            print()
        cmm3=sentence1.feedback(cmm3,mmm2_predicted)
        
        #input(cmm3[1,:,:])
        #print(thresholded_cm_3_motor_added)
        #print(' above is thresholded_cm_3_motor_added\n' )
        #print(connection_matrix_3_motor_added)
        #print('above is connection_matrix_3_motor added. all connection matrix motors for ON sdr 3 bits are added together to form one memory matrix2\n')
        #print(connection_matrix_3_MOTOR)
        #print('above is connection_matrix_3_MOTOR\n')
        #print(connection_matrix_3_motor_added*sentence1.memory_matrix_2)
        #print('\n above is connection_matrix_3_motor_added*sentence1.memory_matrix_2 \n')
        #print(sentence1.memory_matrix_2)
        #print('\n above is memory_matrix_2 \n')
        '''
        print(sentence1.sdr_matrix_3)
        print('\n aobe is SDR matrix 3 \n ')
        count=count+1
    
            
if __name__ == '__main__':
  main()
