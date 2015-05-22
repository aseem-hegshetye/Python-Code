
import numpy
import pprint
import sys
import operator
import math
import time
class SENTENCE2():
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    memory_matrix_rows=10
    memory_matrix_columns=200 #INCREASE THREHOLD WITH NUMBER OF COLUMNS. HIGH THRESH ARE GOOD FOR SEMANTIC SIMILARITY IN NEXT PREDICTIONS
    ascii_matrix_columns=10  # its the length of binary input. i have written a code to set it up. so dont change this number, its not flexible
    onbit_ascii=2 #number of ON bits in every row of ascii matrix
    percentage_of_on_bits_sdr=8
    threshold=2  #Maximum connection strength .. CM also CM33
    synapse_mat_threshold=2 # SYNAPSE MATRIX CONNECTIONS NEED TO BE LESS STRONG, SO ASS TO ENCOURAGE NOVEL PREDICTIONS EVEN WHEN FEW BITS OVERLAP
    percent_CM_motor=50 #percentage of ON connections per SDR bit in Connection Matrix Motor
    
    no_on_bits_sdr=math.floor(percentage_of_on_bits_sdr*memory_matrix_columns/100) #THIS THING IS OUTDATED AND ONLY USED IN MMMOTOR OR COMMENTS LOL
    on_bits_sdr=no_on_bits_sdr
    #NEW
    on_bits_sdr3=8
    on_bits_sdr2=7          #NEW  NEEDS TO BE CHANGED WITH COLUMNS IF U WISH
    on_bits_sdr1=5
    
    max_bit_CM_motor_added=no_on_bits_sdr*threshold
    threshold_CMM_added=max_bit_CM_motor_added*0.3 # 30% of max stregnth of a bit in cmm_added.
    #So the added cmm has to have atleast 50% of the max magnitude to be considered as 1
    cmm3_rows=10 # connection matrix motor 3 rows
        
    min_num_ofsdr_bits=math.ceil(no_on_bits_sdr*-1) # minimum number of bits in SDR that should be 1 in order for that sdr to be considered worth and valid
    num_on_sdr3=3 # number of on bits in SDR3 per input pattern. if there are two words in a SENTENCE2 there will be 2*2 on bits in SDR3
    max_rows_sdr1_down=20 #max 0 rows that the sdr1 down mat is initiated with
    
    
    def feedback(self,cmm3,mmm2_predicted): #adjusts cmm3 to get mmm3_predicted = mmm3
        corection_matrix=self.mmm2-mmm2_predicted
        for i in range(0,SENTENCE2.memory_matrix_columns*SENTENCE2.memory_matrix_rows):
            if corection_matrix.flat[i]==1:
                for bit in range(0,SENTENCE2.memory_matrix_columns):
                    if self.sdr3_motor[bit]==1:
                        if cmm3[bit,:,:].flat[i]<1:
                            cmm3[bit,:,:].flat[i]=1
                            #if cmm3[bit,:,:].flat[i]<0.8:
                                #cmm3[bit,:,:].flat[i]=cmm3[bit,:,:].flat[i]+0.2
                                #print(' 0.2 addition')
                            #else:
                                #cmm3[bit,:,:].flat[i]=1
                                #print(' 0.2 addition made 1')
                        #elif numpy.isnan(cmm3[bit,:,:].flat[i])== True:
                            #cmm3[bit,:,:].flat[i]=1
                            #print('nan converted to 0.2')

            elif corection_matrix.flat[i]==-1:
                for bit in range(0,SENTENCE2.memory_matrix_columns):
                    if self.sdr3_motor[bit]==1:
                        if (cmm3[bit,:,:].flat[i])<=0.2:
                            cmm3[bit,:,:].flat[i]=numpy.nan
                            #print('made nan')
                        elif (cmm3[bit,:,:].flat[i])>0.2 and (cmm3[bit,:,:].flat[i])<0.8: # dont touch a connection stronger than 0.8. u will loose learnt pattern
                            cmm3[bit,:,:].flat[i]=cmm3[bit,:,:].flat[i]-0.2
                            #print(' 0.2 substraction')
                        
                        

        return(cmm3)

    def Predicting_MMM2(self,cmm3):
        #~~ predicting mmm2
        mmm2_added=self.Connection_matrix_sum_of_OnSdr(self.sdr3_motor,cmm3) #~cm of all ON sdr bits are added
        #input(cmm3[1,:,:])
        #cmm3_added_thresholded=SENTENCE2.thresholding_of_matrix(cmm3_added,SENTENCE2.threshold_CMM_added)# all bits above threshold are kept as it is and others made 0
        for row in range(0,SENTENCE2.cmm3_rows): # choosing top bits per row.
            mmm2_added[row,:]=SENTENCE2.top_bit(mmm2_added[row,:],SENTENCE2.no_on_bits_sdr)# top bits are set to 1
            if sum (mmm2_added[row,:])<SENTENCE2.no_on_bits_sdr: # if less top bits than required to represent a patern then make whole row 0
                mmm2_added[row,:]=0
        return(mmm2_added)

    def SDR3_MOTOR(self,cmm3):
        cmm3_zero=SENTENCE2.replacing_nan_with_zero(cmm3)
        self.sdr3_motor = numpy.zeros((SENTENCE2.memory_matrix_columns))
        
        for bit in range(0,SENTENCE2.memory_matrix_columns):
            self.sdr3_motor[bit]=sum(sum((cmm3_zero[bit,:,:]*(self.mmm2[:,:]))))
            
        self.sdr3_motor[:]=SENTENCE2.top_bit(self.sdr3_motor[:],SENTENCE2.on_bits_sdr)

        #   strengthening CMM 3
        
        for bitt in range(0,SENTENCE2.memory_matrix_columns):
            if self.sdr_matrix_3[bitt]==1:
                cmm3[bitt,:,:]=cmm3[bitt,:,:]+((self.mmm2[:,:])) #cmm3[bitt,:,:]=cmm3[bitt,:,:]+((self.mmm2[:,:])/5)
                #input(cmm3[bitt,:,:])
                #for testing, adding 1 directly instead of 0.2
                #print(' added 1 of mmm2 to cmm3 in SDR3_motor function !!')
                    #keeping it below threshold
                for coll in range(0,SENTENCE2.memory_matrix_columns):
                    for rrw in range(0,SENTENCE2.memory_matrix_rows):
                        if cmm3[bitt,rrw,coll]> SENTENCE2.threshold:
                            cmm3[bitt,rrw,coll]=SENTENCE2.threshold
                            #input('\nthreshold in SDR matrix 3 reached\n')
                #input(cmm3[bitt,:,:])
        #input(cmm3[30,:,:])
        return(cmm3)



    def MMM2(self):
        self.mmm2=numpy.zeros((SENTENCE2.cmm3_rows,SENTENCE2.memory_matrix_columns))
        for row in range(0,(self.sdr_matrix_2).shape[0]):
            self.mmm2[row,:]=self.sdr_matrix_2[row,:]*1
    
    def CMM3(cmm3): #BUILDING CMM3
        '''for bitt in range(0,SENTENCE2.memory_matrix_columns):
            if self.sdr_matrix_3[bitt]==1:
                for row in range (0,(self.sdr_matrix_2).shape[0]):
                    cmm3[bitt,row,:]=cmm3[bitt,row,:]+(self.sdr_matrix_2[row,:]/5)
                
        cmm3=SENTENCE2.thresholding_of_matrix_2(cmm3,SENTENCE2.threshold) #> threshold is set = to threshold
        '''
        matrx=cmm3[0:math.ceil(.5*SENTENCE2.memory_matrix_columns),0:math.ceil(.25*SENTENCE2.cmm3_rows),:]
        matrx=SENTENCE2.setting_50perct_nan(matrx)
        cmm3[0:math.ceil(.5*SENTENCE2.memory_matrix_columns),math.ceil(.25*SENTENCE2.cmm3_rows):,:]=numpy.nan

        matrx=cmm3[math.ceil(.5*SENTENCE2.memory_matrix_columns):math.ceil(.75*SENTENCE2.memory_matrix_columns),math.ceil(.25*SENTENCE2.cmm3_rows):math.ceil(.50*SENTENCE2.cmm3_rows),:]
        matrx=SENTENCE2.setting_50perct_nan(matrx)
        cmm3[math.ceil(.5*SENTENCE2.memory_matrix_columns):math.ceil(.75*SENTENCE2.memory_matrix_columns),0:math.ceil(.25*SENTENCE2.cmm3_rows),:]=numpy.nan
        cmm3[math.ceil(.5*SENTENCE2.memory_matrix_columns):math.ceil(.75*SENTENCE2.memory_matrix_columns),math.ceil(.50*SENTENCE2.cmm3_rows):,:]=numpy.nan

        matrx=cmm3[math.ceil(.75*SENTENCE2.memory_matrix_columns):math.ceil(.90*SENTENCE2.memory_matrix_columns),math.ceil(.50*SENTENCE2.cmm3_rows):math.ceil(.75*SENTENCE2.cmm3_rows),:]
        matrx=SENTENCE2.setting_50perct_nan(matrx)
        cmm3[math.ceil(.75*SENTENCE2.memory_matrix_columns):math.ceil(.90*SENTENCE2.memory_matrix_columns),0:math.ceil(.50*SENTENCE2.cmm3_rows),:]=numpy.nan
        cmm3[math.ceil(.75*SENTENCE2.memory_matrix_columns):math.ceil(.90*SENTENCE2.memory_matrix_columns),math.ceil(.75*SENTENCE2.cmm3_rows):,:]=numpy.nan


        matrx=cmm3[math.ceil(.90*SENTENCE2.memory_matrix_columns):,math.ceil(.75*SENTENCE2.cmm3_rows):,:]
        matrx=SENTENCE2.setting_50perct_nan(matrx)
        cmm3[math.ceil(.90*SENTENCE2.memory_matrix_columns):,0:math.ceil(.75*SENTENCE2.cmm3_rows),:]=numpy.nan
        
        #input(cmm3[49,:,:])
        return(cmm3)

        

    def Memory_Matrix_1(self,synapse_matrix_one,inhibition_mat):
        self.memory_matrix_1=numpy.zeros((self.number_of_words,SENTENCE2.memory_matrix_rows,SENTENCE2.memory_matrix_columns))
        
        pred_available=self.memory_matrix_1[0,1:,:]+1 # it has all bits that are aavailable to represent a pattern. as bits get used they become 0.so when
        #multiplied with prediction matrix we are left with bits that r free for new pattern representation.. damn!!
        #EVERY BIT IN MEMORY MATRIX IS A CELL. IT CAN ONLY REPRESENT ONE PATTERN AT A TIME FOR ONE SDR REPRESENTATION
        for wrd in range(0,self.number_of_words):
            
            pred_available=(pred_available*0)+1 # refreshing the whole matrix.,making it all 1
            
            self.memory_matrix_1[wrd,0,:]=self.sdr_matrix_1[wrd,0,:]*1
            new_memory_matrix=self.memory_matrix_1[wrd,:,:]*0
            #print(self.sdr_matrix_1)
            #print('above was sdr matrix 1 ' )
            #print(self.memory_matrix_1)
            #print('above was mem matrix 1 in function Memory_Matrix_1\n' )
            history_matrix=self.memory_matrix_1[wrd,:,:]*1
            for rw in range(1,self.len_of_longestword):
                prediction_matrix=SENTENCE2.Next_prediction(history_matrix,synapse_matrix_one)
                #history_matrix=history_matrix*0    
                for col in range(0,SENTENCE2.memory_matrix_columns):
                    if self.sdr_matrix_1[wrd,rw,col]>0:
                        max_pred=numpy.argmax(prediction_matrix[1:,col]*pred_available[:,col])+1 #IT GIVES MAX PREDICTED VALUE IN THAT COLUMN TO MAKE IT 1
                        self.memory_matrix_1[wrd,max_pred,col]=1
                        new_memory_matrix[max_pred,col]=1
                #print (new_memory_matrix)
                #print('above is new_memory_matrix 1 pattern ')
                pred_available=pred_available-new_memory_matrix[1:,:]# USED BITS ARE MADE 0 IN PRED_AVAILABLE. SO THEY ARE NO LONGER AVAILABLE
                       
                synapse_matrix_one=SENTENCE2.strengthening_synapse(synapse_matrix_one,history_matrix,new_memory_matrix)
                #print (history_matrix)
                #print('above is history_matrix \n SYNAPSE ARE BEING STRENGTHENED AND INHIBITION TOO\n ')
                #print (new_memory_matrix)
                #input('above is new_memory_matrix ')
                
                #STRENGTHENING INHIBITION MATRIX~~ no new function needed. old function INHIBITION_MATRIX2 does the same
                inhibition_mat=SENTENCE2.INHIBITION_MATRIX2(inhibition_mat,new_memory_matrix[1:,:])
            
                history_matrix=new_memory_matrix*1
                new_memory_matrix=new_memory_matrix*0
        return(synapse_matrix_one,inhibition_mat)

    def Memory_Matrix_2(self,synapse_matrix_two,inhibition_mat):
        self.memory_matrix_2=numpy.zeros((SENTENCE2.memory_matrix_rows,SENTENCE2.memory_matrix_columns))
        #input('executing memory matrix 2 ')
        
        pred_available=self.memory_matrix_2[1:,:]+1 # ALL 1 MATRIX
        
        self.memory_matrix_2[0,:]=self.sdr_matrix_2[0,:]*1
        new_memory_matrix=self.memory_matrix_2[:,:]*0
        #print(self.sdr_matrix_2)
        #input('above was sdr matrix 2 ' )
        #print(self.memory_matrix_2)
        #input('above was mem matrix 2 ' )
        history_matrix=self.memory_matrix_2[:,:]*1
        for rw in range(1,self.number_of_words):
            
            prediction_matrix=SENTENCE2.Next_prediction(history_matrix,synapse_matrix_two)
            #history_matrix=history_matrix*0    
            for col in range(0,SENTENCE2.memory_matrix_columns):
                if self.sdr_matrix_2[rw,col]>0:
                    max_pred=numpy.argmax(prediction_matrix[1:,col]*pred_available[:,col])+1
                    self.memory_matrix_2[max_pred,col]=1
                    new_memory_matrix[max_pred,col]=1

            pred_available=pred_available-new_memory_matrix[1:,:]# USED BITS ARE MADE 0 IN PRED_AVAILABLE. SO THEY ARE NO LONGER AVAILABLE
                    
            synapse_matrix_two=SENTENCE2.strengthening_synapse(synapse_matrix_two,history_matrix,new_memory_matrix)
            #print(new_memory_matrix[1:,:])
            #input('above is new pattern to connect inhibition matrix\n')
            
            #STRENGTHENING INHIBITION MATRIX ~~~~
            inhibition_mat=SENTENCE2.INHIBITION_MATRIX2(inhibition_mat,new_memory_matrix[1:,:])
            
            #print(new_memory_matrix[1:,:])
            #print('above is new memory mat\n')
            #input(inhibition_mat)
            history_matrix=new_memory_matrix*1
            new_memory_matrix=new_memory_matrix*0
        return(synapse_matrix_two,inhibition_mat)		
		


    def strengthening_synapse(synapse_matrix,history_matrix,memory_matrix):
        m=SENTENCE2.memory_matrix_rows
        n=SENTENCE2.memory_matrix_columns
        
        for j in range(0,(m*n)):
            if memory_matrix.flat[j]>0:
                for i in range(0,(m*n)):
                    if history_matrix.flat[i]>0:
                        # OLD PATCH .. ADDING +1
                        #synapse_mat_threshold is SMALLER THEN NORMAL THRESHOLD. COZ OTHER BITS SHOULD HAVE POWER OF PREDICTING NOVEL PATTERNS. OTHERWISE IF PATTERN
                        #OVERLAP WITH 3 BITS, THAN THIS 3 BITS WILL STROGLY PREDICT THERE STRONGER CONNECTIONS AND OTHER 3-4 ON BITS WONT BE ABLE TO INFLUENCE THE PATTERN
                        #IF MORE THAN 3 BITS OVERLAP. GOD SAVE ME
                        temp=synapse_matrix[i,:,:]
                        #if i!=j:
                        if (temp.flat[j]+100) >SENTENCE2.synapse_mat_threshold:
                            temp.flat[j]=SENTENCE2.synapse_mat_threshold
                        else:
                            temp.flat[j]=temp.flat[j]+100
                            input('100 added to synapse matrix :O ?? how \n')
                            
                        synapse_matrix[i,:,:]=temp
                        
                        '''
                     #strengthening the whole column with next cell
                        (row,col)=SENTENCE2.row_column_from_flatinput(m,n,i) # row is 0-m... column is 0-n		
                        for r in range(0,m):
                            i=(r*n)+col
                            temp=synapse_matrix[i,:,:]
                        #if i!=j:
                            if temp.flat[j] <SENTENCE2.threshold:
                                temp.flat[j]=temp.flat[j]+0.2
                                synapse_matrix[i,:,:]=temp
                        '''
                     
                        
        return (synapse_matrix)

    def Next_prediction(history_matrix,synapse_matrix):
	# RETURNS A "MATRIX" (it may have more than required columns active and also more cells per column active.. !! loads of prediction)
	#OF NEXT PREDICTION.
        trial2=history_matrix*0
        for i in range(0,SENTENCE2.memory_matrix_rows*SENTENCE2.memory_matrix_columns):
            if history_matrix.flat[i]>0:
                trial2=trial2+synapse_matrix[i,:,:]
        return trial2

    def Sdr_Matrix_3(self,connection_matrix_3,cm3_down):
        #build SDR matrix 3
        connection_matrix_zero_3=SENTENCE2.replacing_nan_with_zero(connection_matrix_3)
        self.sdr_matrix_3 = numpy.zeros((SENTENCE2.memory_matrix_columns))
        
        for bit in range(0,SENTENCE2.memory_matrix_columns):
            self.sdr_matrix_3[bit]=sum(sum((connection_matrix_zero_3[bit,:,:]*(self.memory_matrix_2[:,:]))))
            #multiplication is weird in python 3.3. one multiplication will give one SDR2 bit. after forming all bits in one row
            # we will take top bits now and then strengthen there connections
        self.sdr_matrix_3_notopbit=self.sdr_matrix_3*1
        
        #OLD PATCH VARIABLE SDR3 BITS
        #self.sdr_matrix_3[:]=SENTENCE2.top_bit(self.sdr_matrix_3[:],(SENTENCE2.num_on_sdr3*(self.sdr_matrix_2.shape[0])))
        #NEW PATCH
        self.sdr_matrix_3[:]=SENTENCE2.top_bit(self.sdr_matrix_3[:],(SENTENCE2.on_bits_sdr3))

        #   strengthening CM3_down
        
        for bitt in range(0,SENTENCE2.memory_matrix_columns):
            if self.sdr_matrix_3[bitt]==1:
                #cm3_down[bitt,:,:]=cm3_down[bitt,:,:]+((self.memory_matrix_2[:,:])/5)
                cm3_down[bitt,:,:]=cm3_down[bitt,:,:]+((self.memory_matrix_2[:,:])*10)
                    #keeping it below threshold
                for coll in range(0,SENTENCE2.memory_matrix_columns):
                    for rrw in range(0,SENTENCE2.memory_matrix_rows):
                        if cm3_down[bitt,rrw,coll]> SENTENCE2.threshold:
                            cm3_down[bitt,rrw,coll]=SENTENCE2.threshold
                            #input('\nthreshold in SDR matrix 3 reached\n')

        ''' #   strengthening CM 3
        
        for bitt in range(0,SENTENCE2.memory_matrix_columns):
            if self.sdr_matrix_3[bitt]==1:
                #connection_matrix_3[bitt,:,:]=connection_matrix_3[bitt,:,:]+((self.memory_matrix_2[:,:])/5)
                connection_matrix_3[bitt,:,:]=connection_matrix_3[bitt,:,:]+((self.memory_matrix_2[:,:]))
                    #keeping it below threshold
                for coll in range(0,SENTENCE2.memory_matrix_columns):
                    for rrw in range(0,SENTENCE2.memory_matrix_rows):
                        if connection_matrix_3[bitt,rrw,coll]> SENTENCE2.threshold:
                            connection_matrix_3[bitt,rrw,coll]=SENTENCE2.threshold
                            #input('\nthreshold in SDR matrix 3 reached\n')
         '''   

        '''
            #   strengthening CM MOTOR 3
        
        for bitt in range(0,SENTENCE2.memory_matrix_columns):
            if self.sdr_matrix_3[bitt]==1:
                connection_matrix_3_motor[bitt,:,:]=connection_matrix_3_motor[bitt,:,:]+((self.memory_matrix_2[:,:])/5)
                    #keeping it below threshold
                for coll in range(0,SENTENCE2.memory_matrix_columns):
                    for rrw in range(0,SENTENCE2.memory_matrix_rows):
                        if connection_matrix_3_motor[bitt,rrw,coll]> SENTENCE2.threshold:
                            connection_matrix_3_motor[bitt,rrw,coll]=SENTENCE2.threshold
                            #input('\nthreshold in SDR matrix 3 reached\n')
        '''
            
        #return(connection_matrix_3,connection_matrix_3_motor)
        return(connection_matrix_3)
        

    def Sdr_Matrix_2(self,connection_matrix_2,cm2_down):
        #build SDR matrix 2
        connection_matrix_zero_2=SENTENCE2.replacing_nan_with_zero(connection_matrix_2)
        self.sdr_matrix_2 = numpy.zeros((self.number_of_words,SENTENCE2.memory_matrix_columns))
        for row in range(0,self.number_of_words):
            for bit in range(0,SENTENCE2.memory_matrix_columns):
                self.sdr_matrix_2[row,bit]=sum(sum((connection_matrix_zero_2[bit,:,:]*(self.memory_matrix_1[row,:,:]))))
                #multiplication is weird in python 3.3. one multiplication will give one SDR2 bit. after forming all bits in one row
                # we will take top bits now and then strengthen there connections
            self.sdr_matrix_2[row,:]=SENTENCE2.top_bit(self.sdr_matrix_2[row,:],SENTENCE2.on_bits_sdr2)

        #   strengthening CM2_down

        for row2 in range(0,self.sdr_matrix_2.shape[0]):
            
            for bitt in range(0,SENTENCE2.memory_matrix_columns):
                if self.sdr_matrix_2[row2,bitt]==1:
                    #cm3_down[bitt,:,:]=cm3_down[bitt,:,:]+((self.memory_matrix_2[:,:])/5)
                    cm2_down[bitt,:,:]=cm2_down[bitt,:,:]+((self.memory_matrix_1[row2,:,:]*10))
                        #keeping it below threshold
                    for coll in range(0,SENTENCE2.memory_matrix_columns):
                        for rrw in range(0,SENTENCE2.memory_matrix_rows):
                            if cm2_down[bitt,rrw,coll]> SENTENCE2.threshold:
                                cm2_down[bitt,rrw,coll]=SENTENCE2.threshold
                                #input('\nthreshold in SDR matrix 3 reached\n')



                #   strengthening CM
            '''
            for bitt in range(0,SENTENCE2.memory_matrix_columns):
                if self.sdr_matrix_2[row,bitt]==1:
                    connection_matrix_2[bitt,:,:]=connection_matrix_2[bitt,:,:]+(self.memory_matrix_1[row,:,:]/5)
                        #keeping it below threshold
                    for coll in range(0,SENTENCE2.memory_matrix_columns):
                        for rrw in range(0,SENTENCE2.memory_matrix_rows):
                            if connection_matrix_2[bitt,rrw,coll]> SENTENCE2.threshold:
                                connection_matrix_2[bitt,rrw,coll]=SENTENCE2.threshold
                '''
        return(connection_matrix_2)

    def ASCII_matrix_from_SDR1(sdr,cm1_down):
        #PLEASE MAKE SURE THAT SDR IS ALWAYS A MATRIX.EVEN IF ITS 1 ROW VECTOR ITS A MATRIX
        #SO SDR[0,:] IS NOT AN ERROR .. PLEASE PLEASE PLEASE . ITS VERY AWFUL AND APALLING
        # ~~ cm1_down is 2D .. 
        ascii_mat=sdr*0
        ascii_mat=ascii_mat[:,:,0:SENTENCE2.ascii_matrix_columns]# ascii_mat will be like the sdr mat except less columns
        #SINCE sdr is always a matrix ( I HAVE CHECKED ) FIRST TWO IF STATEMENTS WONT EVER EXECUTE
        #SO ITS SAFE TO KEEP SDR_ROW EVEN THOUGH SUCH VARIABLE DOESNT EXIST !!
        if len(sdr.shape)==1:
            ascii_row=(sdr_row*cm1_down).sum(axis=1) # one ascii row
        elif len(sdr.shape)==2:
            for row in range(0,sdr.shape[0]):
                ascii_row[row,:]=(sdr_row[row,:]*cm1_down).sum(axis=1) # one ascii row
        elif len(sdr.shape)==3:
            for third_dim in range(0,sdr.shape[0]):
                for row in range(0,sdr.shape[1]):
                    vector=(sdr[third_dim,row,:]*cm1_down).sum(axis=1) # one ascii row
                    ascii_mat[third_dim,row,:]=SENTENCE2.top_bit(vector,SENTENCE2.onbit_ascii)
        return(ascii_mat)
        

    def Sdr_Matrix_1(self,connection_matrix_1,cm1_down):
        #build SDR matrix 1
        connection_matrix_zero_1=SENTENCE2.replacing_nan_with_zero(connection_matrix_1)
        #print('connection matrix replaced NAN with zero')
        #print(connection_matrix_zero_1)
        self.sdr_matrix_1 = numpy.zeros((self.number_of_words,self.len_of_longestword,SENTENCE2.memory_matrix_columns))
        #there will atleast be 1 word.. so sdr_matrix_1 will have atleast 1 in all 3 dimensions :D so u r free to use 3D indices.
        # ITS APPALING ~~!!!!
        #print('SRD_matrix_1 initialized to zero')
        #print(self.sdr_matrix_1)
        for wrd in range(0,self.number_of_words):
            for row in range(0,self.len_of_longestword):
                
                self.sdr_matrix_1[wrd,row,:]=sum((connection_matrix_zero_1*(self.ascii_matrix[wrd,row,:])).T)
                #multiplication is weird in python 3.3
                self.sdr_matrix_1[wrd,row,:]=SENTENCE2.top_bit(self.sdr_matrix_1[wrd,row,:],SENTENCE2.on_bits_sdr1)

        
                                
        #   strengthening CM1_down
        for third_dim in range(0,self.sdr_matrix_1.shape[0]):
            for row in range(0,self.sdr_matrix_1.shape[1]):
                for bit in range(0,SENTENCE2.ascii_matrix_columns):
                    if self.ascii_matrix[third_dim,row,bit]==1:
                        cm1_down[bit,:]=cm1_down[bit,:]+(self.sdr_matrix_1[third_dim,row,:]*10)

                        #KEEPING BELOW THRESHOLD
                        for coll in range(0,SENTENCE2.memory_matrix_columns):
                            if cm1_down[bit,coll]> SENTENCE2.threshold:
                                    cm1_down[bit,coll]=SENTENCE2.threshold
                

        

        #print('sdr_matrix_1 ')
        #print(self.sdr_matrix_1)
        #print('connection_matrix_zero_1')
        #print(connection_matrix_zero_1)
        #print('self.ascii_matrix')
        #print(self.ascii_matrix)
        return(connection_matrix_1)
            
        

    def setting_50perct_nan(mat):
        #setting 50% of connections to NAN
        if len(mat.shape)==2:
            for i in range(0,mat.shape[0]):
                for count in range(0,numpy.int(math.floor(0.5*mat.shape[1]))):
                    while True:
                        rand_num=numpy.random.random_integers(0,mat.shape[1]-1)
                        if (numpy.isnan(mat[i,rand_num]))== False:
                    #if that connection is not already a NAN then set it as NAN and also increase the count, else just continue iterations without increasing the count
                            mat[i,rand_num]=numpy.nan
                            break
                                    
        elif len(mat.shape)==3:
            for third_dimension in range(0,mat.shape[0]):
                for i in range(0,mat.shape[1]):
                    for count in range(0,numpy.int(math.floor(0.5*mat.shape[2]))):
                        while True:
                            rand_num=numpy.random.random_integers(0,mat.shape[2]-1)
                            if (numpy.isnan(mat[third_dimension,i,rand_num]))== False:
                                mat[third_dimension,i,rand_num]=numpy.nan
                                break
                                    
        return (mat)
        
    
    def Connection_Matrix_1(connection_matrix):
        #setting 50% of connections to NAN
        
        for i in range(0,SENTENCE2.memory_matrix_columns):
            for count in range(0,numpy.int(math.floor(0.5*SENTENCE2.ascii_matrix_columns))):
                while True:
                    rand_num=numpy.random.random_integers(0,SENTENCE2.ascii_matrix_columns-1)
                    if (numpy.isnan(connection_matrix[i,rand_num]))== False:
		#if that connection is not already a NAN then set it as NAN and also increase the count, else just continue iterations without increasing the count
                        connection_matrix[i,rand_num]=numpy.nan
                        break
				
        return (connection_matrix)

    def Connection_Matrix_2(connection_matrix):
        for third_dimension in range(0,SENTENCE2.memory_matrix_columns):
            for i in range(0,SENTENCE2.memory_matrix_rows):
                for count in range(0,numpy.int(math.floor(0.5*SENTENCE2.memory_matrix_columns))):
                    while True:
                        rand_num=numpy.random.random_integers(0,SENTENCE2.memory_matrix_columns-1)
                        if (numpy.isnan(connection_matrix[third_dimension,i,rand_num]))== False:
                            connection_matrix[third_dimension,i,rand_num]=numpy.nan
                            break
				
        return (connection_matrix)
				    
    def __init__(self):
        self.number_of_words=0 # number of words in a SENTENCE2    
        
    
    def RemovePunctuations(self,my_str):
        #also converts it to lower case
        self.my_str=my_str
        # remove punctuations from the string
        self.no_punct = ""
        for char in my_str:
           if char not in SENTENCE2.punctuations:
               self.no_punct = self.no_punct + char
        # conver to lower case and display the unpunctuated string
        self.no_punct_lower=self.no_punct.lower()
        #print(self.no_punct_lower)
        return (self.no_punct_lower)
    
    def SeparateWords(self,my_str):
        #self.my_str=my_str
        #print('words separated')
        #print(my_str.split())
        return my_str.split()

    def NumberofWords(self,my_str):
        self.words=my_str  #['how','are','you']
        self.number_of_words= (len(self.words))
        #print(self.words)
        #input()
        self.len_of_longestword=len(max(self.words,key=len))
        #print (self.len_of_longestword)
        #input('len of max word\n')
        

    def Length_of_each_word(self,k):
        self.length_of_each_word=[0]*self.number_of_words
        for i in range (0,self.number_of_words):
            self.length_of_each_word[i]=len(list(k[i]))

    def Ascii_Matrix(self):
        #self.ascii_matrix = [[[0 for k in range(SENTENCE2.ascii_matrix_columns)] for j in range(self.len_of_longestword)] for i in range(self.number_of_words)]
        self.ascii_matrix = numpy.zeros((self.number_of_words,self.len_of_longestword,SENTENCE2.ascii_matrix_columns))
        #pprint.pprint (self.ascii_matrix )
        for i in range(0,self.number_of_words):
            word=list(self.words[i])  # it has the current word under consideration
            for j in range(0,self.length_of_each_word[i]):
                bin_character=self.Converting_char_2binary(word[j])
                self.ascii_matrix[i,j,:]=bin_character
        #print('ascii matrix \n')
        #pprint.pprint (self.ascii_matrix )    

    def CHARACTERS_OUTPUT(self):
        #it give characters from self.ascii_mat_down.
        #self.ascii_mat_down is always a matrix!! (I have checked). feel free to index it.
        output=[]
        
        if len(self.ascii_mat_down.shape)==1:# 1 letter
            if sum(self.ascii_mat_down)==2:
                output.append(SENTENCE2.Converting_binary_2char(self.ascii_mat_down))
        elif len(self.ascii_mat_down.shape)==2:# 1 word
            
            for row in range(0,self.ascii_mat_down.shape[0]):
                if sum(self.ascii_mat_down[row,:])==2:
                    output.append(SENTENCE2.Converting_binary_2char(self.ascii_mat_down[row,:]))
        elif len(self.ascii_mat_down.shape)==3:# many words 1 SENTENCE2
            #print('in CHARACTERS_OUTPUT function ascii mat is 3D\n')
            for third_dim in range(0,self.ascii_mat_down.shape[0]):
                for row in range(0,self.ascii_mat_down.shape[1]):
                    if sum(self.ascii_mat_down[third_dim,row,:])==2:
                        output.append(SENTENCE2.Converting_binary_2char(self.ascii_mat_down[third_dim,row,:]))
                output.append(' ')

        return(output)
                    

        
    def Converting_binary_2char(bin_vect): #mke sure bin_vect is a vector with 1st dimension as number of columns
        m=-1
        
        #print(bin_vect,'\n this is the input ascii row to be converted into character\n')
        for bit in range(0,SENTENCE2.ascii_matrix_columns):
            if bin_vect[bit]==1:
                if m<0: # m not set. so this 1 bit is the first bit
                    m=bit*1
                    #print('setting m=',m)
                else:
                    n=bit*1
                    #print('\nsetting n=',n,' \n now breaking the for loop')
                    break
        #print(bin_vect,'\n this is the input ascii row to be converted into character\n','m= ',m,'\n n= ',n )
        if m==0:
            decimal=97+n-1
        elif m==1:
            decimal=106+n-2
        elif m==2:
            decimal=114+n-3
        elif m==3:
            decimal=121+n-4
        return(chr(decimal))

    def Converting_char_2binary(self,x):
        #x is a char input . FUNCTION TAKES A CHAR AND RETURNS ITS BINARY THAT I MADE
        y=[0]*SENTENCE2.ascii_matrix_columns
        if ord(x)>=97 and ord(x)<=105:
            m=0
            n=97
        elif ord(x)>=106 and ord(x)<=113:
            m=1
            n=106
        elif ord(x)>=114 and ord(x)<=120:
            m=2
            n=114
        elif ord(x)>=121 and ord(x)<=124:
            m=3
            n=121
        y[m]=1
        #print (y)
        #print((ord(x)),n,m)
        y[(ord(x)-n)+m+1]=1
        #print (y)
        return y


    def replacing_nan_with_zero(connection_matrix):
        matrx=connection_matrix+(connection_matrix*0)
	#PYTHON DOESNT HAVE PASS BY VALUE, AND I DONT WANT IT TO MODULATTE CONNECTION MATRIX BY REPLACING NAN. 
	#THIS NEW MATRX IS RETURNED AND IS STORED IN CONNECTION_MATRIX_ZERO
        if len(matrx.shape)==3:
            for i in range(0,matrx.shape[0]):
                for j in range(0,matrx.shape[1]):
                    for k in range(0,matrx.shape[2]):
                        if numpy.isnan(matrx[i,j,k]):
                            matrx[i,j,k]=0
        elif len(matrx.shape)==2:
            for i in range(0,matrx.shape[0]):
                for j in range(0,matrx.shape[1]):
                    if numpy.isnan(matrx[i,j]):
                        matrx[i,j]=0
                                    
        return matrx

    def top_bit(vect_in,num):
        #Vect_in can be a 1D or 2D.. if num of positive bits dont exist no problem. it will only consider positive number and make them 1
        temp= vect_in*0
        
        vect=vect_in+(vect_in*0)
        #CREATED A NEW OBJECT SO THAT THE PASSED ORIGINAL VECTOR IS UNCHANGED
        for count in range(0,num):
            if numpy.max(vect)>0:
                index_to_change=numpy.argmax(vect)
                #temp[index_to_change]=vect[index_to_change]
                temp.flat[index_to_change]=1   # SETS TOP BITS 1
                vect.flat[index_to_change]=0
        return (temp)


    def row_column_from_flatinput(numberofrows,numberofcolumns,flatinput): # row is 0-m... column is 0-n
        for r in range (0,numberofrows):
            for c in range(0,numberofcolumns):
                if flatinput==(r*numberofcolumns)+c:
                    row = r
                    column = c
        return (row,column)

    def Connection_matrix_sum_of_OnSdr(self,sdr_vector,cm):
        cm_zero=SENTENCE2.replacing_nan_with_zero(cm)
        cm_added=cm_zero[0,:,:]*0
        for i in range(0,SENTENCE2.memory_matrix_columns):
            if sdr_vector[i]>0:
                cm_added=cm_added+cm_zero[i,:,:]
                #print('connections added')
        return(cm_added)

    def pruning(cm1):
        #it turns lower percent_CM_motor percent bits in Connection matrix (cm) to 0
        #also substract 0.1 from all connections to keep it live !! ;)
        cm=cm1*1
        cm_new=cm*0
        for col in range(0,SENTENCE2.memory_matrix_columns):
            for count in range(0,math.floor(percent_CM_motor*SENTENCE2.memory_matrix_columns*SENTENCE2.memory_matrix_rows/100)):
                if sum(sum(cm[col,:,:]))==0:
                    break
                else:
                    indx=numpy.argmax(cm[col,:,:])
                    cm_new[col,:,:].flat[indx]=(cm[col,:,:].flat[indx])-0.1
                    cm[col,:,:].flat[indx]=0

        return(cm_new)

    def thresholding_of_matrix(matrx,thresh):
        #all >= threshold as it is.. < threshold = 0
        mat=matrx*0
        if len(matrx.shape)==3:
            for i in range(0,matrx.shape[0]):
                for j in range(0,matrx.shape[1]):
                    for k in range(0,matrx.shape[2]):
                        if (matrx[i,j,k])>=thresh:
                            mat[i,j,k]=matrx[i,j,k]
        elif len(matrx.shape)==2:
            for i in range(0,matrx.shape[0]):
                for j in range(0,matrx.shape[1]):
                    if (matrx[i,j])>=thresh:
                        mat[i,j]=matrx[i,j]

        elif len(matrx.shape)==1:
            for i in range(0,matrx.shape[0]):
                if (matrx[i])>=thresh:
                    mat[i]=matrx[i]

        return(mat)

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

        return(mat)


    def PREDICTING_MM2(self,connection_matrix_3): #normal prediction
        
        #mm2_added=self.Connection_matrix_sum_of_OnSdr(self.sdr_matrix_3_notopbit,connection_matrix_3) #~cm of all ON sdr bits are added
        #input(cmm3[1,:,:])
        #cmm3_added_thresholded=SENTENCE2.thresholding_of_matrix(cmm3_added,SENTENCE2.threshold_CMM_added)# all bits above threshold are kept as it is and others made 0
        '''
        for row in range(0,SENTENCE2.cmm3_rows): # choosing top bits per row.
            mmm2_added[row,:]=SENTENCE2.top_bit(mmm2_added[row,:],SENTENCE2.no_on_bits_sdr)# top bits are set to 1
            if sum (mmm2_added[row,:])<SENTENCE2.no_on_bits_sdr: # if less top bits than required to represent a patern then make whole row 0
                mmm2_added[row,:]=0
        '''
        cm_zero=SENTENCE2.replacing_nan_with_zero(connection_matrix_3)
        cm_added=cm_zero[0,:,:]*0
        for i in range(0,SENTENCE2.memory_matrix_columns):
            #if self.sdr_matrix_3_notopbit[i]>0:
            if self.sdr_matrix_3[i]==1:
                #cm_added=cm_added+(cm_zero[i,:,:]*self.sdr_matrix_3_notopbit[i])
                cm_added=cm_added+(cm_zero[i,:,:])
                #print('connections added')
        return(cm_added)

    def PREDICTING_MM(self,connection_matrix_2): #normal prediction
        
        cm_zero=SENTENCE2.replacing_nan_with_zero(connection_matrix_2)
        #print(len(self.sdr2_down.shape))
        #print('above is lenth of sdr2_down dimensions\n')
        if len(self.sdr2_down.shape)==1:
            third_dimension=1
        else:
            third_dimension=self.sdr2_down.shape[0]
            
        #print(self.sdr2_down.shape)
        #input(' shape of sdr2_down in PREDICTING_MM talk support2 begnining \n' )
        cm_added=numpy.zeros((third_dimension,SENTENCE2.memory_matrix_rows,SENTENCE2.memory_matrix_columns))
        
        # !!! making sure prog doesnt stick if only 1 word is i/p.. it give too many indices error
        if len(self.sdr2_down.shape)>1: 
            for row in range(0,self.sdr2_down.shape[0]):
                for i in range(0,SENTENCE2.memory_matrix_columns):
                    #if self.sdr_matrix_3_notopbit[i]>0:
                    if self.sdr2_down[row,i]==1:
                        #cm_added=cm_added+(cm_zero[i,:,:]*self.sdr_matrix_3_notopbit[i])
                        cm_added[row,:,:]=cm_added[row,:,:]+(cm_zero[i,:,:])
                        #print('connections added')
        elif len(self.sdr2_down.shape)==1:
            #print(' problem 1 solved\n' )
            for i in range(0,SENTENCE2.memory_matrix_columns):
                    if self.sdr2_down[i]==1:
                        cm_added[:,:]=cm_added[:,:]+(cm_zero[i,:,:])
        #print(self.sdr2_down.shape)
        #input(' shape of sdr2_down in PREDICTING_MM talk support2 end\n' )
        #print(cm_added.shape)
        #input(' shape of cm_added in PREDICTING_MM talk support2 end\n' )
        return(cm_added)
            
    def Predicted_or_not(mm2_pred,mm2):
        for i in range(0,mm2.shape[0]*mm2.shape[1]):
            if mm2.flat[i]==1 and mm2_pred.flat[i]!=10:
                #print(' \nnot predicted completly')
                break
            if mm2_pred.flat[i]==10 and mm2.flat[i]!=1:
                input('\nECTRA predicted')
            elif i== (mm2.shape[0]*mm2.shape[1])-1:
                print(' \nyahoo PREDICTED !!' )
                
    def INHIBITION_MATRIX2_1strow(inhibition_mat,sdr_row): # forming connections
        for i in range(0,SENTENCE2.memory_matrix_columns):
            if sdr_row[i]==1:
                for j in range(0,SENTENCE2.memory_matrix_columns):
                    if sdr_row[j]==1 and j!=i:
                        inhibition_mat[i,j]=1
                        inhibition_mat[j,i]=1 #both are on simultaneously. so strengthen both with each other unlike synapse mat :p

        return (inhibition_mat)

    def INHIBITION_MATRIX1_1strow(self,inhibition_mat): # forming connections
        for dim in range(0,self.sdr_matrix_2.shape[0]):
            
            for i in range(0,SENTENCE2.memory_matrix_columns):
                if self.sdr_matrix_1[dim,0,i]==1:
                    for j in range(0,SENTENCE2.memory_matrix_columns):
                        if self.sdr_matrix_1[dim,0,j]==1 and j!=i:
                            inhibition_mat[i,j]=1
                            inhibition_mat[j,i]=1 #both are on simultaneously. so strengthen both with each other unlike synapse mat :p

        return (inhibition_mat)

    def INHIBITION_MATRIX2(inhibition_mat,matrx): # forming inhibitory connections for whole mem mat except 1st row
        #matrx is the memory matrix excluding the first row
        for i in range(0,(matrx.shape[0]*matrx.shape[1])):
            if matrx.flat[i]>0:
                for j in range(0,(matrx.shape[0]*matrx.shape[1])):
                    if matrx.flat[j]>0 and j!=i:
                        
                        
                        inhibition_mat[i,:,:].flat[j]=1
                        inhibition_mat[j,:,:].flat[i]=1 #both are on simultaneously. so strengthen both with each other unlike synapse mat :p
                        #print(inhibition_mat[i,:,:])
                        #print(inhibition_mat[j,:,:])
                        #input('making inhibition 1\n')

        return (inhibition_mat)
    

    def INHIBITED_MM2_1strow(self,mm2_pred,inhibition_mat):# this function is independant of learning. it will be directly called while predicting
        
        final_inhibited=mm2_pred[0,:]*0
        self.on_bit_sdr3=SENTENCE2.NUMBER_OF_ON_BITS(self.sdr_matrix_3) # NUMBER OF ON BITS ARE FIXED IN SDR3 AS IN EVERY SDR, BUT STILL M CALCULATING IT HERE AGAIN :P
        #JUST INCASE IF THERE IS ERROR.. WHICH CANNOT HAPPEN COZ I M GOING TO CHOOSE TOP BITS FROM PREDICTED SDR3 LOL
        for col in range(0,SENTENCE2.memory_matrix_columns):
            if mm2_pred[0,col]>math.floor((self.on_bit_sdr3*SENTENCE2.threshold)*0.9): # (on_bit_sdr3*SENTENCE2.threshold) - is max bit strength.. uska 80% liya
                final_inhibited=final_inhibited+(inhibition_mat[col]*mm2_pred[0,col])

        #on_bit_sdr3=SENTENCE2.NUMBER_OF_ON_BITS(self.sdr_matrix_3) #GIVES NUMBER OF ON BITS IN SDR 3
        mm1strow_pred_tresh=SENTENCE2.thresholding_of_matrix(mm2_pred[0,:],(self.on_bit_sdr3*SENTENCE2.threshold)*0.9) #only above thresholds are kept.others 0
        
        #making it 1 or 0
                #OLD PATCH
        '''
        for ind in range(0,(final_inhibited.shape[0])):
            if final_inhibited[ind]>=0:
                final_inhibited[ind]=1
            else:
                final_inhibited[ind]=0
        '''
        # NEW PATCH
        final_inhibited=SENTENCE2.top_bit(final_inhibited*mm1strow_pred_tresh,SENTENCE2.on_bits_sdr2)
        
        self.sdr2_down=final_inhibited
        return(final_inhibited)

    def INHIBITED_MM1_1strow(self,mm1_pred,inhibition_mat):# this function is independant of learning. it will be directly called while predicting
        
        #final_inhibited=mm1_pred[:,0,:]*0 # its for 3D memory matrix 1st row
        final_inhibited=numpy.zeros((mm1_pred.shape[0],SENTENCE2.memory_matrix_columns))
        #print(final_inhibited.shape,' final_inhibited shape in begining of imhibited_mm1_1st row\n')
        for third_dim in range(0,mm1_pred.shape[0]):
            
            #print((mm1_pred.shape))
            #input(' shape of mm1_pred in support talk2 inhibited_mm1_1strow\n')
            # !!!! making sure prog doesnt stick if only 1 word is i/p.. it give too many indices error
            # dimension of mm1_pred are (1,rows,columns) !!!!!!!!!!!!!!!!!!!!
            if  mm1_pred.shape[0]<=1: 
                on_bit_sdr2=SENTENCE2.NUMBER_OF_ON_BITS(self.sdr2_down[:])
            else:
                on_bit_sdr2=SENTENCE2.NUMBER_OF_ON_BITS(self.sdr2_down[third_dim,:])

            #print(mm1_pred[third_dim,0,:],'\n above is mm1_pred first row\n')
            #print((on_bit_sdr2*SENTENCE2.threshold)*0.6)
            mm1strow_pred_tresh=SENTENCE2.thresholding_of_matrix(mm1_pred[third_dim,0,:],(on_bit_sdr2*SENTENCE2.threshold)*0.99) #only above thresholds are kept.others 0
            #print(mm1strow_pred_tresh,'\n above is mm1strow_pred_tresh \n')
            for col in range(0,SENTENCE2.memory_matrix_columns):
                if mm1_pred[third_dim,0,col]>math.floor((on_bit_sdr2*SENTENCE2.threshold)*0.99): # (on_bit_sdr3*SENTENCE2.threshold) - is max bit strength.. uska 80% liya
                    #print(final_inhibited.shape,'\nfinal inhibited shape')
                    #print(mm1_pred.shape,'\nm1_pred shape')
                    
                    final_inhibited[third_dim,:]=final_inhibited[third_dim,:]+(inhibition_mat[col]*mm1_pred[third_dim,0,col])

            #making it 1 or 0
                    #OLD PATCH
            '''
            print(final_inhibited,'\n final_inhibited before making it 1 or 0\n')
            for ind in range(0,(final_inhibited.shape[1])):
                if (final_inhibited[third_dim,:]*mm1strow_pred_tresh).flat[ind]>0:
                    final_inhibited[third_dim,ind]=1
                else:
                    final_inhibited[third_dim,ind]=0
            '''
            #NEW PATCH
            final_inhibited[third_dim,:]=SENTENCE2.top_bit(final_inhibited[third_dim,:]*mm1strow_pred_tresh,SENTENCE2.on_bits_sdr1)

            #print('\n final_inhibited after making it 1 or 0\n')
            #input(final_inhibited)
            
                    
        self.sdr1_1strow_down=final_inhibited
        return(final_inhibited)

    def INHIBITED_MM2(self,mm2_pred,inhibition_mat,inhibited_mm2_1strow,synapse_matrix_2):
        
        a=numpy.zeros((SENTENCE2.memory_matrix_rows-1,SENTENCE2.memory_matrix_columns))
        final_inhibited=a*0
        zero_1strow=numpy.zeros((1,SENTENCE2.memory_matrix_columns))
        
        on_bits_prev_pattern=SENTENCE2.NUMBER_OF_ON_BITS(inhibited_mm2_1strow)
        inhb_mat=numpy.vstack((inhibited_mm2_1strow,a))*1 # i dont want mm2 1st row to change. so *1
        prediction_matrix=SENTENCE2.Next_prediction(inhb_mat,synapse_matrix_2)
        mm2_pred_tresh=SENTENCE2.thresholding_of_matrix(mm2_pred,(self.on_bit_sdr3*SENTENCE2.threshold)*0.99) #only above thresholds are kept.others 0
        
        #HERE THRESHOLD SHOULD BE TIGHT..!! SO IT DOESNT PREDICT UNNECESARRY BITS AND CAUSE CAOS BELOW
        
        while True:
            
            if sum(sum(mm2_pred_tresh))==0:
                break
            #print(mm2_pred_tresh)
            #print('mm2_pred_tresh \n')
            final_inhibited=a*0
            pred_mat_tresh=SENTENCE2.thresholding_of_matrix(prediction_matrix,(on_bits_prev_pattern*SENTENCE2.synapse_mat_threshold)*0.99) #>thresholds = threshold, others 0
            #WE HAVE A SEPARATE synapse_mat_threshold NOW WHICH IS LOWER THAN NORMAL threshold
            
            mult=mm2_pred_tresh*pred_mat_tresh
            if sum(sum(mult))==0:
                break
            
            #inhibition
            for index in range(0,(SENTENCE2.memory_matrix_rows-1)*SENTENCE2.memory_matrix_columns):
                if mult[1:,:].flat[index]>0: 
                    final_inhibited=final_inhibited+(inhibition_mat[index,:,:]*mult[1:,:].flat[index])

            #MAKING it 1 or 0
            #OLD PATCH
            '''
            for ind in range(0,(final_inhibited.shape[0]*final_inhibited.shape[1])):
                if final_inhibited.flat[ind]>=0:
                    final_inhibited.flat[ind]=1
                else:
                    final_inhibited.flat[ind]=0
            '''
            #NEW PATCH
            final_inhibited=SENTENCE2.top_bit(final_inhibited*mult[1:,:],SENTENCE2.on_bits_sdr2)
            
            #print(final_inhibited)
            #print('final_inhibited it has 1 or 0 \n')
            if sum(sum(final_inhibited))<SENTENCE2.min_num_ofsdr_bits:
                break

            #storing new pattern in SDR2
            final_ivector=numpy.zeros((final_inhibited.shape[1]))
            for col in range(0,SENTENCE2.memory_matrix_columns):
                if sum(final_inhibited[:,col])>0:
                    final_ivector[col]=1
            #print(final_ivector)
            #input('above is final_ivector\n' )
            self.sdr2_down=numpy.vstack((self.sdr2_down,final_ivector ))

            on_bits_prev_pattern=SENTENCE2.NUMBER_OF_ON_BITS(final_ivector) #mostly its 5, but later we may change number of on bits according to the size
                
            inhb_mat=numpy.vstack((zero_1strow,final_inhibited))*1 # i dont want mm2 1st row to change. so *1
            #print(inhb_mat)
            #input('inhb_mat inhibited matrix\n')
            prediction_matrix=SENTENCE2.Next_prediction(inhb_mat,synapse_matrix_2)

        
            
        return(final_inhibited)

    def INHIBITED_MM1(self,mm1_pred,inhibition_mat,inhibited_mm1_1strow,synapse_matrix_1):
        # if ONE LETTER IS NOT PREDICTED ADEQUATLY IN A WORD. IT JUST BREAKS AND MOVES ON TO THE NEXT WORD PREDICTION :(
        
        a=numpy.zeros((SENTENCE2.memory_matrix_rows-1,SENTENCE2.memory_matrix_columns))
        final_inhibited=a*0
        zero_1strow=numpy.zeros((1,SENTENCE2.memory_matrix_columns))
        self.sdr1_down=numpy.zeros((self.sdr2_down.shape[0],SENTENCE2.max_rows_sdr1_down,SENTENCE2.memory_matrix_columns))
        #self.sdr_matrix_2 is a 2D mat.fuck off. so shape[0] will atleast be 1
        #print(self.sdr1_down.shape,'self.sdr1_down shape after its initialization\n')
        max_count_sdr1down_rows=1

        if len(self.sdr2_down.shape)==1:
            county=1
        else:
            county=self.sdr2_down.shape[0]
        
        for third_dim in range(0,county):
            on_bits_prev_pattern=SENTENCE2.NUMBER_OF_ON_BITS(inhibited_mm1_1strow[third_dim])
            inhb_mat=numpy.vstack((inhibited_mm1_1strow[third_dim],a))*1 # i dont want mm2 1st row to change. so *1
            prediction_matrix=SENTENCE2.Next_prediction(inhb_mat,synapse_matrix_1)
            
            #!!!! fixing patch for one word input
            if  mm1_pred.shape[0]<=1: 
                on_bit_sdr2=SENTENCE2.NUMBER_OF_ON_BITS(self.sdr2_down[:])
            else:
                on_bit_sdr2=SENTENCE2.NUMBER_OF_ON_BITS(self.sdr2_down[third_dim,:])
                
            #on_bit_sdr2=SENTENCE2.NUMBER_OF_ON_BITS(self.sdr2_down[third_dim,:])
            
            
            #THRESHOLDS HERE SHOULD BE LOW COZ UPPER MATRIX MAY MAKE PREDICTION MISTAKES. THIS MAT NEEDS TO BE OPEN TO ERRORS.
            mm1_pred_tresh=SENTENCE2.thresholding_of_matrix(mm1_pred[third_dim,:,:],(on_bit_sdr2*SENTENCE2.threshold)*0.99) #only above thresholds are kept.others 0
            mm1_tomakezero=mm1_pred_tresh[1:,:]*1
            sdr1_down=self.sdr1_1strow_down[third_dim,:]*1 #1st row of that dim memory mat is stored. its bloody 1D vector as of now.
            
            while True:
                #print(' in while loop of INHIBITED_MM1')
                if sum(sum(mm1_tomakezero))==0:
                    #print('\n mm1_tomakezero == 0.. so all predicted paterns have been stored in SDR matrix. BREAK NOW\n')
                    break
                #print(mm1_pred_tresh)
                #print('above is mm1_pred_tresh \n')
                final_inhibited=a*0
                #print(prediction_matrix)
                #print(' prediction_matrix \n')
                #print(on_bits_prev_pattern)
                #input('on_bits_prev_pattern .. i think it should be 5 fixed.. on bits per letter\n')
                pred_mat_tresh=SENTENCE2.thresholding_of_matrix(prediction_matrix,(on_bits_prev_pattern*SENTENCE2.synapse_mat_threshold)*0.99) #>=thresholds = threshold, others 0
                #WE HAVE A SEPARATE synapse_mat_threshold NOW WHICH IS LOWER THAN NORMAL threshold

                mult=mm1_pred_tresh*pred_mat_tresh # MAY HAVE OVERLAPPING BITS.. SO NOT TAKING MM1_TOMAKEZERO HERE

                
                    
                if (SENTENCE2.NUMBER_OF_ON_BITS(pred_mat_tresh))<SENTENCE2.min_num_ofsdr_bits:
                    #print(mm1_pred_tresh)
                    #print('above is mm1_pred_tresh \n')
                    #print(prediction_matrix.astype(int),'\n above is prediction_matrix\n')
                    #print(pred_mat_tresh.astype(int),'\n above is pred_mat_tresh')
                    #print(mult.astype(int))
                    #print(' above is multiplication.. mm1 pred thresh * pred_mat_tresh\n')
                    #input(' pred_mat_tresh doesnt have enough ON bits .. so BREAK\n above is all DETAILS U require\n')
                    break
                #print(pred_mat_tresh)
                #print(' above is pred_mat_tresh \n')
                
                if sum(sum(mult))==0:
                    #print(' mul is zero .. so break \n')
                    break
                
                #inhibition
                for index in range(0,(SENTENCE2.memory_matrix_rows-1)*SENTENCE2.memory_matrix_columns):
                    if mult[1:,:].flat[index]>0: 
                        final_inhibited=final_inhibited+(inhibition_mat[index,:,:]*mult[1:,:].flat[index])
                #print(final_inhibited)
                #input('inhibition matrix\n')
    
                #---------------------------
                #ERROR CHECKING PATCH
                '''
                print(prediction_matrix.astype(int),'\n above is prediction_matrix\n')
                print(pred_mat_tresh.astype(int),'\n above is pred_mat_tresh')
                print(mult.astype(int))
                print(' above is multiplication.. mm1 pred thresh * pred_mat_tresh\n')
                print(final_inhibited,'\nabove is final_inhibited matrix. it has some ZHOL i guess\n')
                print(final_inhibited*mult[1:,:],'\nfinal_inhibited matrix *mult[1:,:]) \n')
                print(inhb_mat,'\n its previous pattern that is predicting above matrix\n')
                #input('printing synapse_matrix_1 of all ON bits in previous pattern\n')
                for bit in range(0,SENTENCE2.memory_matrix_columns*SENTENCE2.memory_matrix_rows):
                    if inhb_mat.flat[bit]==1:
                        print(synapse_matrix_1[bit,:,:].astype(int))
                #input( 'printed all synapse matrix of on bits\n')
                
                for index in range(0,(SENTENCE2.memory_matrix_rows-1)*SENTENCE2.memory_matrix_columns):
                    if mult[1:,:].flat[index]>0:
                        print(inhibition_mat[index,:,:])
                #input('above are inhibition matrices that are added for final inhibition\n')
                
                '''
                #-----------------------------------------

                #making it 1 or 0
                #BIGGEST SCAMSTER ~~!!
                        #OLD PATCH
                '''
                for ind in range(0,(final_inhibited.shape[0]*final_inhibited.shape[1])):
                    if (final_inhibited*mult[1:,:]).flat[ind]>0:
                        final_inhibited.flat[ind]=1
                    else:
                        final_inhibited.flat[ind]=0
                '''

                    #NEW PATCH
                final_inhibited=SENTENCE2.top_bit(final_inhibited*mult[1:,:],SENTENCE2.on_bits_sdr1)
                
                #print(final_inhibited)
                #input('final_inhibited it has 1 or 0 \n')
                if sum(sum(final_inhibited))<SENTENCE2.min_num_ofsdr_bits: # FINAL INHIBITED IS FOR PARTICULAR PREDICTION MATRIX
                    #print(' num of on bits=',sum(sum(final_inhibited)),'\nnum of ON bits in SDR1 is less.. so BREAK\n')
                    #print(mm1_pred_tresh)
                    #print('above is mm1_pred_tresh \n')
                    #print(final_inhibited)
                    #input('final_inhibited it has 1 or 0 \n')
                    break
                
                #REMOVING ALL BITS FROM MM1_PRED_THRESH THAT ARE IN FINAL INHIBITED. SO NOW ONLY REMAINING ON BITS NEED TO BE PREDICTED
                #ONCE ALL BBITS ARE PREDICTED FROM MM1_PRED_THRESH , BREAK !!
                mm1_tomakezero=((final_inhibited*-1)+1)*mm1_tomakezero

                #storing new pattern in SDR1
                final_ivector=numpy.zeros((SENTENCE2.memory_matrix_columns))
                for col in range(0,SENTENCE2.memory_matrix_columns):
                    if sum(final_inhibited[:,col])>0:
                        final_ivector[col]=1
                #print(final_ivector)
                #print('above is final_ivector\n' )
                sdr1_down=numpy.vstack((sdr1_down,final_ivector ))

                on_bits_prev_pattern=SENTENCE2.NUMBER_OF_ON_BITS(final_ivector) #mostly its 5, but later we may change number of on bits according to the size
                    
                inhb_mat=numpy.vstack((zero_1strow,final_inhibited))*1 # i dont want mm2 1st row to change. so *1
                #print(inhb_mat)
                #input('inhb_mat inhibited matrix\n')
                prediction_matrix=SENTENCE2.Next_prediction(inhb_mat,synapse_matrix_1)
                

            # STORING SDR1_DOWN-2D IN SELF.SDR1DOWN-3D
            
            if len(sdr1_down.shape)>1: # if there are more than 1 rows.. store each row of sdr1_down in each row of self.sdr1_down
                for roww in range(0,sdr1_down.shape[0]):
                    #print(self.sdr1_down.shape)
                    #print(' SELF.sdr1_down shape.. its the final SDR1 matrix\n')
                    #print(sdr1_down.shape)
                    #input(' sdr1_down shape. its temporary 2 dimension matrix\n')
                    #print(sdr1_down)
                    #print(' above is sdr1_down. its to be stored in self.sdr1_down which is 3D \n' )
                    self.sdr1_down[third_dim,roww,:]=sdr1_down[roww,:]*1
            elif len(sdr1_down.shape)==1: # if it just has 1 row.. then directly store it in its first row
                try:
                    #print('shape of self.sdr1_down = ',self.sdr1_down.shape,'\n shape of sdr1_down= ',sdr1_down.shape )
                    #input('this line had an error previously\n give keyboard interrupt if u want to see mm1_pred_thresh and predictions\n')
                    self.sdr1_down[third_dim,0,:]=sdr1_down*1
                except:
                    #print(mm1_pred_tresh)
                    #print('above is mm1_pred_tresh \n')
                    #print(prediction_matrix,'\n above is prediction_matrix \n')
                    print('')
                    #print(final_inhibited)
                    #print('final_inhibited it has 1 or 0. (final_inhibited *mult)>0 =1 \n')
                    
                    

            #CALCULATING THE MAX WORD LENGTH PREDICTED. SO THAT IS THE MAX ROWS OF SDR1
            if sdr1_down.shape[0]> max_count_sdr1down_rows and len(sdr1_down.shape)>1:
                max_count_sdr1down_rows=sdr1_down.shape[0]*1
                #print(max_count_sdr1down_rows)
                #print(' max_count_sdr1down_rows \n' )

        self.sdr1_down=self.sdr1_down[:,0:max_count_sdr1down_rows,:]

            
            
        
    

    def NUMBER_OF_ON_BITS(vect):
        count=0
        if len(vect.shape)==1:
            for ind in range(0,vect.shape[0]):
                if vect[ind]>0:
                   count=count+1
        if len(vect.shape)==2:
            for ind1 in range(0,vect.shape[0]):
                for ind2 in range(0,vect.shape[1]):
                    if vect[ind1,ind2]>0:
                       count=count+1

        return(count)
    
    def inhibition_mat_zeroing_own_dimension(mat):
        #it sets connection of a bit with itself as zero.. so it doesnt inhibit itself.
        if len(mat.shape)==3:
            for i in range(0,mat.shape[0]):
                mat[i,:,:].flat[i]=0
        elif len(mat.shape)==2:
            for j in range(0,mat.shape[0]):
                mat[j,j]=0
        return(mat)
    
    def CM33(cm33,sdr3_1,sdr3_2):
        #print(sdr3_2.shape[0],'\n its a 1D vector preety sure\n')
            
        for i in range(0,sdr3_2.shape[0]):
            if sdr3_2[i]==1:
                cm33[i,:]=cm33[i,:]+(sdr3_1 *10)

        cm33=SENTENCE2.thresholding_of_matrix_2(cm33,SENTENCE2.threshold)
            
        return(cm33)

    
        
        
        
'''print(no_punct.split())
k=no_punct.split()
print(list(k[0]))
print(list(k[1]))
''' 


        
