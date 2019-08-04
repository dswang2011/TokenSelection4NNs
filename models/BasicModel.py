
# -*- coding: utf-8 -*-
from keras import optimizers
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, Input, model_from_json, load_model, Sequential
from keras import backend as K
from keras.layers import Layer,Dense
from models.matching import Attention,getOptimizer,precision_batch,identity_loss,MarginLoss,Cosine
class BasicModel(object):
    def __init__(self,opt): 
        self.opt=opt
        self.model = self.get_model(opt)
        self.model.compile(optimizer=optimizers.Adam(lr=opt.lr), loss='categorical_crossentropy', metrics=['acc'])
        
    def get_model(self,opt):

        return None

    
    def train(self,train,dev=None,dirname="saved_model"):
        x_train,y_train = train
        
        
        filename = os.path.join( dirname,  "best_model_" + self.__class__.__name__+".h5" )
        callbacks = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath=filename, monitor='val_loss', save_best_only=True,save_weights_only=True)]
        if dev is None:
            history = self.model.fit(x_train,y_train,batch_size=self.opt.batch_size,epochs=self.opt.epoch_num,callbacks=callbacks,validation_split=self.opt.validation_split,shuffle=True)
        else:
            x_val, y_val = dev
            history = self.model.fit(x_train,y_train,batch_size=self.opt.batch_size,epochs=self.opt.epoch_num,callbacks=callbacks,validation_data=(x_val, y_val),shuffle=True) 
        os.rename(filename,os.path.join( dirname,  str(max(history.history["val_acc"])) +"_" + self.__class__.__name__+"_"+self.opt.to_string()+".h5" ))
        
       
    def predict(self,x_test):
        return self.model.predict(x_test)
    
    def save(self,filename="model",dirname="saved_model"):
        filename = os.path.join( dirname,filename + "_" + self.__class__.__name__ +".h5")
        self.model.save(filename)
        return filename
    
    def get_pair_model(self):
        representation_model = self.model
        representation_model.pop()
        self.question = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
        self.answer = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
        self.neg_answer = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
        
        if self.opt.match_type == 'pointwise':
            reps = [representation_model(doc) for doc in [self.question, self.answer]]
            output = Dense(self.opt.nb_classes, activation="softmax")(Attention()(reps))
           
#         ,
            model = Model([self.question,self.answer], output)
            model.compile(loss = "categorical_hinge",  optimizer = getOptimizer(name=self.opt.optimizer,lr=self.opt.lr), metrics=["acc"])
            
        elif self.opt.match_type == 'pairwise':

            q_rep = representation_model(self.question)

            score1 = Cosine([q_rep, representation_model.get_representation(self.answer)])
            score2 = Cosine([q_rep, representation_model.get_representation(self.neg_answer)])
            basic_loss = MarginLoss(self.opt.margin)([score1,score2])
            
            output=[score1,basic_loss,basic_loss]
            model = Model([self.question, self.answer, self.neg_answer], output) 
            model.compile(loss = identity_loss,optimizer = getOptimizer(name=self.opt.lr.optimizer,lr=self.opt.lr), 
                          metrics=[precision_batch],loss_weights=[0.0, 1.0,0.0])
        
    
    def train_matching(self,train,dev=None,dirname="saved_model"):
        self.model =  self.get_pair_model(self.opt)
        self.train(train,dev=dev,dirname=dirname)










