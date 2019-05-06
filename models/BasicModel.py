
# -*- coding: utf-8 -*-
from keras import optimizers
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint

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
             ModelCheckpoint(filepath=filename, monitor='val_loss', save_best_only=True)]
        if dev is None:
            history = self.model.fit(x_train,y_train,batch_size=self.opt.batch_size,epochs=self.opt.epoch_num,callbacks=callbacks,validation_split=self.opt.validation_split)
        else:
            x_val, y_val = dev
            history = self.model.fit(x_train,y_train,batch_size=self.opt.batch_size,epochs=self.opt.epoch_num,callbacks=callbacks,validation_data=(x_val, y_val)) 
        os.rename(filename,os.path.join( dirname,  str(min(history.history["val_loss"])) +"_" + self.__class__.__name__+"_"+self.opt.to_string()+".h5" ))
        
       
    def predict(self,x_test):
        return self.model.predict(x_test)
    
    def save(self,filename="model",dirname="saved_model"):
        filename = os.path.join( dirname,filename + "_" + self.__class__.__name__ +".h5")
        self.model.save(filename)
        return filename
    
