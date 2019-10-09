import numpy as np
import pandas as pd
import os
import re
from keras import applications
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from keras.utils import Sequence
from keras.utils import np_utils
from keras import backend as K

'''
Parameters
    - name:  model name without prefix or suffix, e.g., cnn5-350
    - artistname:  e.g., 'Rembrandt'
    - codepath: typically,
        'C:/Users/Steve/Google Drive/MOOCs/Computer Science Resources/AI/AI and Art/Code/'
    - trainpath:  to directory containing artist and 'other' tiles, e.g.,
        'D:/A-Eye/Rembrandt/data/train400'
    - validatepath:  e.g.,
        'D:/A-Eye/Rembrandt/data/validate400'
    -  modelpath:  where model is sourced or stored, e.g., same as codepath
        'C:/Users/Steve/Google Drive/MOOCs/Computer Science Resources/AI/AI and Art/Code/'
'''

class cnn3_1D(object):
    def __init__(self, img_size, name, artistname, codepath, modelpath=None, new=False, batch_size=16, epochs=50, patience=5, save_every_epoch=False):
        self.img_size = img_size
        self.codepath = codepath        
        self.modelpath = modelpath
        self.name = name
        self.artistname = artistname
        self.batch_size = batch_size
        self.new = new
        self.epochs = epochs
        self.patience = patience
        self.save_every_epoch = save_every_epoch
        if self.new:
            self.model = cnn3_1D.makeModel(self.img_size, self.name)
        else:
            self.model = load_model(self.modelpath+self.name+'.h5')
            print("Model "+name+" loaded")
            
    def makeModel(img_size, name):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(img_size, img_size, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(50, (3, 3))) #for 50 pixels only, otherwise 64
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(50)) #for 50 pixels only, otherwise 64
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        print("Model "+name+" created or instantiated.\n")
        return model
    
    def train(self, trainpath, validatepath):
        model = self.model
        nb_train_samples = sum(len(files) for _, _, files in os.walk(trainpath))
        nb_validation_samples = sum(len(files) for _, _, files in os.walk(validatepath))
        epochs = self.epochs
        batch_size = self.batch_size
        if self.save_every_epoch:            
            mc = ModelCheckpoint(filepath=self.codepath+"saved-2Dmodel-"+str(self.img_size)+"-{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}.h5", \
                    monitor='val_acc', verbose=1, save_best_only=False, mode='max')
            callbacks_list = [mc]
        else:
            es = EarlyStopping(monitor='val_acc', mode='min', verbose=1, patience=self.patience) 
            mc = ModelCheckpoint(filepath=self.codepath+"best_2D-"+self.name+".h5", \
                    monitor='val_acc', verbose=1, save_best_only=True)
            callbacks_list = [es, mc]
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                            horizontal_flip=False)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(
                trainpath,
                target_size=(self.img_size, self.img_size),
                batch_size=self.batch_size,
                class_mode='binary')
        validation_generator = test_datagen.flow_from_directory(
                validatepath,
                target_size=(self.img_size, self.img_size),
                batch_size=batch_size,
                class_mode='binary')
        print("Training "+self.name+" with tiles in "+trainpath)
        model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples // batch_size,
                epochs=epochs,
                callbacks=callbacks_list,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples // batch_size)
        print(self.name+" trained successfully.\n")
        return
    
    def test(self, testpath, spreadsheetname, filewrite=True):
        model = self.model
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
                directory=testpath,
                target_size=(self.img_size, self.img_size),
                color_mode="rgb",
                batch_size=1,
                class_mode='binary',
                shuffle=False,    
                )
        print("Testing "+self.name+" with tiles in "+testpath)
        filenames = test_generator.filenames
        nb_samples = len(filenames)
        test_generator.reset()
        preds = model.predict_generator(test_generator,nb_samples)
        filename_list = []
        for n in filenames:
            filename_list.append(re.search('(.+?)_', n).group(1))
        preds_list = np.squeeze(preds).tolist()
        file_dict = {}
        names = set(filename_list)
        FPs_avg = []
        FNs_avg = []
        FPs_MV = []
        FNs_MV = []
        #corrects = []
        variances = []
        for name in names:
            tilelist = [preds_list[i] for i in range(len(preds_list)) if filename_list[i] == name]
            yes = 0
            FN = 0
            FP = 0
            for tile in tilelist:
                if (self.artistname in name and not "Not" in name and tile < .5):
                    FN += 1
                elif ((self.artistname not in name or "Not" in name) and tile >= .5):
                    FP += 1
                else:
                    yes += 1
            if yes >= (FN+FP):
                maj_vote = "Y"
            else:
                maj_vote = "N"
                if FP>FN:
                    FPs_MV.append(1)
                else:
                    FNs_MV.append(1)
            MVfrac = yes/(yes+FN+FP)
            avg = np.mean(tilelist)
            var = np.var(tilelist)
            variances.append(var)            
            if (self.artistname in name and not "Not" in name and avg < .5):
                FNs_avg.append(1)
                file_dict[name] = (round(avg,6), round(var,6), "N", len(tilelist), maj_vote, MVfrac)
            elif ((self.artistname not in name or "Not" in name) and avg >= .5):
                FPs_avg.append(1)
                file_dict[name] = (round(avg,6), round(var,6), "N", len(tilelist), maj_vote, MVfrac)
            else:
                #corrects.append(1)
                file_dict[name] = (round(avg,6), round(var,6), "Y", len(tilelist), maj_vote, MVfrac)
        tot_incorrect_avg = sum(FPs_avg + FNs_avg)    
        tot_correct_avg = len(names) - tot_incorrect_avg
        tot_incorrect_MV = sum(FPs_MV + FNs_MV)
        tot_correct_MV = len(names) - tot_incorrect_MV
        if filewrite:
            f = open(self.codepath+spreadsheetname+".csv", "a+")
            f.write("\n\nModel: "+self.name)
            f.write("\n"+str(self.img_size)+"-pixel tiles\n")
            f.write("\nName,Average,Variance,Correct?,# tiles,Maj Vote,MV Frac Correct\n")
            for k in file_dict.keys():
                f.write(k+","+str(file_dict[k][0])+","+str(file_dict[k][1])+","+str(file_dict[k][2])+","+str(file_dict[k][3])+" tiles,"+file_dict[k][4]+","+str(file_dict[k][5])+"\n")
            f.write("\nAccuracy by Average: "+str(tot_correct_avg)+" / "+str(tot_incorrect_avg+tot_correct_avg)+" = "+str(round(1.*tot_correct_avg/(tot_correct_avg+tot_incorrect_avg+.00001),2)))
            f.write("\nAccuracy by Majority Vote: "+str(tot_correct_MV)+" / "+str(tot_incorrect_MV+tot_correct_MV)+" = "+str(round(1.*tot_correct_MV/(tot_correct_MV+tot_incorrect_MV+.00001),2)))            
            f.write("\nMean variance: "+str(1.*sum(variances)/len(variances)))
            f.write("\n\n"+str(sum(FPs_avg))+" False Positives by Average\n")
            f.write(str(sum(FNs_avg))+" False Negatives by Average\n")
            f.write("\n"+str(sum(FPs_MV))+" False Positives by Majority Vote\n")
            f.write(str(sum(FNs_MV))+" False Negatives by Majority Vote\n")
            f.close()
        return round(1.*tot_correct_avg/(tot_correct_avg+tot_incorrect_avg+.00001),2), \
            round(1.*tot_correct_MV/(tot_correct_MV+tot_incorrect_MV+.00001),2), \
            1.*sum(variances)/len(variances)
    
    def saveModel(self):
        self.model.save(self.codepath+self.name+'h5')

    def getName(self):
        return self.name

    def getModel(self):
        return self.model
        
    def getModelPath(self):
        return self.modelpath
    
    def getImageSize(self):
        return self.img_size
    
    def __str__(self):
        return 'Model name is '+self.name
    

class cnn5_0D(object):
    def __init__(self, img_size, name, artistname, codepath, modelpath=None, new=False, batch_size=16, epochs=50, patience=5, save_every_epoch=False):
        self.img_size = img_size
        self.codepath = codepath        
        self.modelpath = modelpath
        self.name = name
        self.artistname = artistname
        self.batch_size = batch_size
        self.new = new
        self.epochs = epochs
        self.patience = patience
        self.save_every_epoch = save_every_epoch
        if self.new:
            self.model = cnn5_2D.makeModel(self.img_size, self.name)
        else:
            self.model = load_model(self.modelpath+self.name+'.h5')
            print("Model "+name+" loaded")
            
    def makeModel(img_size, name):
        model = Sequential()
        model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(img_size, activation='relu'))  #a hyperparameter
        model.add(Dense(1, activation = 'sigmoid'))
        print("Model "+name+" created or instantiated.\n")
        return model
    
    def train(self, trainpath, validatepath, l_rate=.001):
        model = self.model
        nb_train_samples = sum(len(files) for _, _, files in os.walk(trainpath))
        nb_validation_samples = sum(len(files) for _, _, files in os.walk(validatepath))
        epochs = self.epochs
        batch_size = self.batch_size
        if self.save_every_epoch:            
            mc = ModelCheckpoint(filepath=self.codepath+"saved-2Dmodel-"+str(self.img_size)+"-{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}.h5", \
                    monitor='val_acc', verbose=1, save_best_only=False, mode='max')
            callbacks_list = [mc]
        else:
            es = EarlyStopping(monitor='val_acc', mode='min', verbose=1, patience=self.patience) 
            mc = ModelCheckpoint(filepath=self.codepath+"best_2D-"+self.name+".h5", \
                    monitor='val_acc', verbose=1, save_best_only=True)
            callbacks_list = [es, mc]
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=l_rate),
                      metrics=['accuracy'])
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                            horizontal_flip=False)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(
                trainpath,
                target_size=(self.img_size, self.img_size),
                batch_size=self.batch_size,
                class_mode='binary')
        validation_generator = test_datagen.flow_from_directory(
                validatepath,
                target_size=(self.img_size, self.img_size),
                batch_size=batch_size,
                class_mode='binary')
        print("Training "+self.name+" with tiles in "+trainpath)
        model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples // batch_size,
                epochs=epochs,
                callbacks=callbacks_list,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples // batch_size)
        print(self.name+" trained successfully.\n")
        return
    
    def test(self, testpath, spreadsheetname, filewrite=True):
        model = self.model
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
                directory=testpath,
                target_size=(self.img_size, self.img_size),
                color_mode="rgb",
                batch_size=1,
                class_mode='binary',
                shuffle=False,    
                )
        print("Testing "+self.name+" with tiles in "+testpath)
        filenames = test_generator.filenames
        nb_samples = len(filenames)
        test_generator.reset()
        preds = model.predict_generator(test_generator,nb_samples)
        filename_list = []
        for n in filenames:
            filename_list.append(re.search('(?<=/)(.+?)_', n).group(1))
			#n1 = re.search('[ \w-]+?(?=\.)', n).group(0)  #for local computer
            #filename_list.append(re.search(r'[^_]*',n1).group(0))
        preds_list = np.squeeze(preds).tolist()
        file_dict = {}
        names = set(filename_list)
        FPs_avg = []
        FNs_avg = []
        FPs_MV = []
        FNs_MV = []
        error = 0
        #corrects = []
        variances = []
        for name in names:
            tilelist = [preds_list[i] for i in range(len(preds_list)) if filename_list[i] == name]
            yes = 0
            FN = 0
            FP = 0
            error_list = []
            for tile in tilelist:
                if (self.artistname in name and not "Not" in name and tile < .5):
                    FN += 1
                    error_list.append(abs(tile - .5))
                elif ((self.artistname not in name or "Not" in name) and tile >= .5):
                    FP += 1
                    error_list.append(abs(tile - .5))
                else:
                    yes += 1
            if yes >= (FN+FP):
                maj_vote = "Y"
            else:
                maj_vote = "N"
                if FP>FN:
                    FPs_MV.append(1)
                else:
                    FNs_MV.append(1)
            if len(error_list) > 0:
                error += sum(error_list)/len(error_list)
            MVfrac = yes/(yes+FN+FP)
            avg = np.mean(tilelist)
            var = np.var(tilelist)
            variances.append(var)            
            if (self.artistname in name and not "Not" in name and avg < .5):
                FNs_avg.append(1)
                file_dict[name] = (round(avg,6), round(var,6), "N", len(tilelist), maj_vote, MVfrac)
            elif ((self.artistname not in name or "Not" in name) and avg >= .5):
                FPs_avg.append(1)
                file_dict[name] = (round(avg,6), round(var,6), "N", len(tilelist), maj_vote, MVfrac)
            else:
                #corrects.append(1)
                file_dict[name] = (round(avg,6), round(var,6), "Y", len(tilelist), maj_vote, MVfrac)
        tot_incorrect_avg = sum(FPs_avg + FNs_avg)    
        tot_correct_avg = len(names) - tot_incorrect_avg
        tot_incorrect_MV = sum(FPs_MV + FNs_MV)
        tot_correct_MV = len(names) - tot_incorrect_MV
        if filewrite:
            f = open(self.codepath+spreadsheetname+".csv", "a+")
            f.write("\n\nModel: "+self.name)
            f.write("\n"+str(self.img_size)+"-pixel tiles\n")
            f.write("\nName,Average,Variance,Correct?,# tiles,Maj Vote,MV Frac Correct\n")
            for k in file_dict.keys():
                f.write(k+","+str(file_dict[k][0])+","+str(file_dict[k][1])+","+str(file_dict[k][2])+","+str(file_dict[k][3])+" tiles,"+file_dict[k][4]+","+str(file_dict[k][5])+"\n")
            f.write("\nAccuracy by Average: "+str(tot_correct_avg)+" / "+str(tot_incorrect_avg+tot_correct_avg)+" = "+str(round(1.*tot_correct_avg/(tot_correct_avg+tot_incorrect_avg+.00001),2)))
            f.write("\nAccuracy by Majority Vote: "+str(tot_correct_MV)+" / "+str(tot_incorrect_MV+tot_correct_MV)+" = "+str(round(1.*tot_correct_MV/(tot_correct_MV+tot_incorrect_MV+.00001),2)))            
            f.write("\nMean variance: "+str(1.*sum(variances)/(len(variances)+.0001)))
            f.write("\n\n"+str(sum(FPs_avg))+" False Positives by Average\n")
            f.write(str(sum(FNs_avg))+" False Negatives by Average\n")
            f.write("\n"+str(sum(FPs_MV))+" False Positives by Majority Vote\n")
            f.write(str(sum(FNs_MV))+" False Negatives by Majority Vote\n")
            f.write("\n\nTotal error = "+str(error)+"\n")
            f.close()
        return round(1.*tot_correct_avg/(tot_correct_avg+tot_incorrect_avg+.00001),2), \
            round(1.*tot_correct_MV/(tot_correct_MV+tot_incorrect_MV+.00001),2), \
            1.*sum(variances)/(len(variances)+.0001)
    
    def testForMap(self, testpath):
        model = self.model
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
                directory=testpath,
                target_size=(self.img_size, self.img_size),
                color_mode="rgb",
                batch_size=1,
                class_mode='binary',
                shuffle=False,    
                )
        print("Testing "+self.name+" with tiles in "+testpath)
        filenames = test_generator.filenames
        nb_samples = len(filenames)
        test_generator.reset()
        preds = model.predict_generator(test_generator,nb_samples)
        filenames=test_generator.filenames
        preds_list = np.squeeze(preds).tolist()
        return filenames, preds_list
    	
    def saveModel(self):
        self.model.save(self.codepath+self.name+'h5')

    def getName(self):
        return self.name

    def getModel(self):
        return self.model
        
    def getModelPath(self):
        return self.modelpath
    
    def getImageSize(self):
        return self.img_size
    
    def __str__(self):
        return 'Model name is '+self.name
    

class cnn5_2D(object):
    def __init__(self, img_size, name, artistname, codepath, modelpath=None, new=False, batch_size=16, epochs=50, patience=5, save_every_epoch=False):
        self.img_size = img_size
        self.codepath = codepath        
        self.modelpath = modelpath
        self.name = name
        self.artistname = artistname
        self.batch_size = batch_size
        self.new = new
        self.epochs = epochs
        self.patience = patience
        self.save_every_epoch = save_every_epoch
        if self.new:
            self.model = cnn5_2D.makeModel(self.img_size, self.name)
        else:
            self.model = load_model(self.modelpath+self.name+'.h5')
            print("Model "+name+" loaded")
            
    def makeModel(img_size, name):
        model = Sequential()
        model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(img_size, activation='relu'))  #a hyperparameter
        model.add(Dropout(0.3))
        model.add(Dense(1, activation = 'sigmoid'))
        print("Model "+name+" created or instantiated.\n")
        return model
    
    def train(self, trainpath, validatepath, l_rate=.001):
        model = self.model
        nb_train_samples = sum(len(files) for _, _, files in os.walk(trainpath))
        nb_validation_samples = sum(len(files) for _, _, files in os.walk(validatepath))
        epochs = self.epochs
        batch_size = self.batch_size
        if self.save_every_epoch:            
            mc = ModelCheckpoint(filepath=self.codepath+"saved-2Dmodel-"+str(self.img_size)+"-{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}.h5", \
                    monitor='val_acc', verbose=1, save_best_only=False, mode='max')
            callbacks_list = [mc]
        else:
            es = EarlyStopping(monitor='val_acc', mode='min', verbose=1, patience=self.patience) 
            mc = ModelCheckpoint(filepath=self.codepath+"best_2D-"+self.name+".h5", \
                    monitor='val_acc', verbose=1, save_best_only=True)
            callbacks_list = [es, mc]
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=l_rate),
                      metrics=['accuracy'])
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                            horizontal_flip=False)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(
                trainpath,
                target_size=(self.img_size, self.img_size),
                batch_size=self.batch_size,
                class_mode='binary')
        validation_generator = test_datagen.flow_from_directory(
                validatepath,
                target_size=(self.img_size, self.img_size),
                batch_size=batch_size,
                class_mode='binary')
        print("Training "+self.name+" with tiles in "+trainpath)
        model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples // batch_size,
                epochs=epochs,
                callbacks=callbacks_list,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples // batch_size)
        print(self.name+" trained successfully.\n")
        return
    
    def test(self, testpath, spreadsheetname, filewrite=True):
        model = self.model
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
                directory=testpath,
                target_size=(self.img_size, self.img_size),
                color_mode="rgb",
                batch_size=1,
                class_mode='binary',
                shuffle=False,    
                )
        print("Testing "+self.name+" with tiles in "+testpath)
        filenames = test_generator.filenames
        nb_samples = len(filenames)
        test_generator.reset()
        preds = model.predict_generator(test_generator,nb_samples)
        filename_list = []
        for n in filenames:
            filename_list.append(re.search('(?<=/)(.+?)_', n).group(1))
			#n1 = re.search('[ \w-]+?(?=\.)', n).group(0)  #for local computer
            #filename_list.append(re.search(r'[^_]*',n1).group(0))
        preds_list = np.squeeze(preds).tolist()
        file_dict = {}
        names = set(filename_list)
        FPs_avg = []
        FNs_avg = []
        FPs_MV = []
        FNs_MV = []
        error = 0
        #corrects = []
        variances = []
        for name in names:
            tilelist = [preds_list[i] for i in range(len(preds_list)) if filename_list[i] == name]
            yes = 0
            FN = 0
            FP = 0
            error_list = []
            for tile in tilelist:
                if (self.artistname in name and not "Not" in name and tile < .5):
                    FN += 1
                    error_list.append(abs(tile - .5))
                elif ((self.artistname not in name or "Not" in name) and tile >= .5):
                    FP += 1
                    error_list.append(abs(tile - .5))
                else:
                    yes += 1
            if yes >= (FN+FP):
                maj_vote = "Y"
            else:
                maj_vote = "N"
                if FP>FN:
                    FPs_MV.append(1)
                else:
                    FNs_MV.append(1)
            if len(error_list) > 0:
                error += sum(error_list)/len(error_list)
            MVfrac = yes/(yes+FN+FP)
            avg = np.mean(tilelist)
            var = np.var(tilelist)
            variances.append(var)            
            if (self.artistname in name and not "Not" in name and avg < .5):
                FNs_avg.append(1)
                file_dict[name] = (round(avg,6), round(var,6), "N", len(tilelist), maj_vote, MVfrac)
            elif ((self.artistname not in name or "Not" in name) and avg >= .5):
                FPs_avg.append(1)
                file_dict[name] = (round(avg,6), round(var,6), "N", len(tilelist), maj_vote, MVfrac)
            else:
                #corrects.append(1)
                file_dict[name] = (round(avg,6), round(var,6), "Y", len(tilelist), maj_vote, MVfrac)
        tot_incorrect_avg = sum(FPs_avg + FNs_avg)    
        tot_correct_avg = len(names) - tot_incorrect_avg
        tot_incorrect_MV = sum(FPs_MV + FNs_MV)
        tot_correct_MV = len(names) - tot_incorrect_MV
        if filewrite:
            f = open(self.codepath+spreadsheetname+".csv", "a+")
            f.write("\n\nModel: "+self.name)
            f.write("\n"+str(self.img_size)+"-pixel tiles\n")
            f.write("\nName,Average,Variance,Correct?,# tiles,Maj Vote,MV Frac Correct\n")
            for k in file_dict.keys():
                f.write(k+","+str(file_dict[k][0])+","+str(file_dict[k][1])+","+str(file_dict[k][2])+","+str(file_dict[k][3])+" tiles,"+file_dict[k][4]+","+str(file_dict[k][5])+"\n")
            f.write("\nAccuracy by Average: "+str(tot_correct_avg)+" / "+str(tot_incorrect_avg+tot_correct_avg)+" = "+str(round(1.*tot_correct_avg/(tot_correct_avg+tot_incorrect_avg+.00001),2)))
            f.write("\nAccuracy by Majority Vote: "+str(tot_correct_MV)+" / "+str(tot_incorrect_MV+tot_correct_MV)+" = "+str(round(1.*tot_correct_MV/(tot_correct_MV+tot_incorrect_MV+.00001),2)))            
            f.write("\nMean variance: "+str(1.*sum(variances)/(len(variances)+.0001)))
            f.write("\n\n"+str(sum(FPs_avg))+" False Positives by Average\n")
            f.write(str(sum(FNs_avg))+" False Negatives by Average\n")
            f.write("\n"+str(sum(FPs_MV))+" False Positives by Majority Vote\n")
            f.write(str(sum(FNs_MV))+" False Negatives by Majority Vote\n")
            f.write("\n\nTotal error = "+str(error)+"\n")
            f.close()
        return round(1.*tot_correct_avg/(tot_correct_avg+tot_incorrect_avg+.00001),2), \
            round(1.*tot_correct_MV/(tot_correct_MV+tot_incorrect_MV+.00001),2), \
            1.*sum(variances)/(len(variances)+.0001)
    
    def testForMap(self, testpath):
        model = self.model
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
                directory=testpath,
                target_size=(self.img_size, self.img_size),
                color_mode="rgb",
                batch_size=1,
                class_mode='binary',
                shuffle=False,    
                )
        print("Testing "+self.name+" with tiles in "+testpath)
        filenames = test_generator.filenames
        nb_samples = len(filenames)
        test_generator.reset()
        preds = model.predict_generator(test_generator,nb_samples)
        filenames=test_generator.filenames
        preds_list = np.squeeze(preds).tolist()
        return filenames, preds_list
    	
    def saveModel(self):
        self.model.save(self.codepath+self.name+'h5')

    def getName(self):
        return self.name

    def getModel(self):
        return self.model
        
    def getModelPath(self):
        return self.modelpath
    
    def getImageSize(self):
        return self.img_size
    
    def __str__(self):
        return 'Model name is '+self.name
    

class cnn5_3D(object):
    def __init__(self, img_size, name, artistname, codepath, modelpath=None, new=False, batch_size=16, epochs=50, patience=5, save_every_epoch=False):
        self.img_size = img_size
        self.codepath = codepath        
        self.modelpath = modelpath
        self.name = name
        self.artistname = artistname
        self.batch_size = batch_size
        self.new = new
        self.epochs = epochs
        self.patience = patience
        self.save_every_epoch = save_every_epoch
        if self.new:
            self.model = cnn5_3D.makeModel(self.img_size, self.name)
        else:
            self.model = load_model(self.modelpath+self.name+'.h5')
            print("Model "+name+" loaded")
            
    def makeModel(img_size, name):
        model = Sequential()
        model.add(Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=(img_size, img_size, 3)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(img_size, activation='relu'))  #a hyperparameter
        model.add(Dropout(0.2))
        model.add(Dense(int(img_size/2), activation='relu'))  #a hyperparameter
        model.add(Dropout(0.3))
        model.add(Dense(1, activation = 'sigmoid'))
        print("Model "+name+" created or instantiated.\n")
        return model
    
    def train(self, trainpath, validatepath):
        model = self.model
        nb_train_samples = sum(len(files) for _, _, files in os.walk(trainpath))
        nb_validation_samples = sum(len(files) for _, _, files in os.walk(validatepath))
        epochs = self.epochs
        batch_size = self.batch_size
        if self.save_every_epoch:            
            mc = ModelCheckpoint(filepath=self.codepath+"saved-3Dmodel-"+str(self.img_size)+"-{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}.h5", \
                    monitor='val_acc', verbose=1, save_best_only=False, mode='max')
            callbacks_list = [mc]
        else:
            es = EarlyStopping(monitor='val_acc', mode='min', verbose=1, patience=self.patience) 
            mc = ModelCheckpoint(filepath=self.codepath+"best_3D-"+self.name+".h5", \
                    monitor='val_acc', verbose=1, save_best_only=True)
            callbacks_list = [es, mc]
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                            horizontal_flip=False)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(
                trainpath,
                target_size=(self.img_size, self.img_size),
                batch_size=self.batch_size,
                class_mode='binary')
        validation_generator = test_datagen.flow_from_directory(
                validatepath,
                target_size=(self.img_size, self.img_size),
                batch_size=batch_size,
                class_mode='binary')
        print("Training "+self.name+" with tiles in "+trainpath)
        model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples // batch_size,
                epochs=epochs,
                callbacks=callbacks_list,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples // batch_size)
        print(self.name+" trained successfully.\n")
        return
    
    def test(self, testpath, spreadsheetname, filewrite=True):
        model = self.model
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
                directory=testpath,
                target_size=(self.img_size, self.img_size),
                color_mode="rgb",
                batch_size=1,
                class_mode='binary',
                shuffle=False,    
                )
        print("Testing "+self.name+" with tiles in "+testpath)
        filenames = test_generator.filenames
        nb_samples = len(filenames)
        test_generator.reset()
        preds = model.predict_generator(test_generator,nb_samples)
        #if filewrite:
        #    results_probs = pd.DataFrame({"Filename":filenames,
        #                  "Predictions":np.squeeze(preds)})
        #    results_probs.to_csv(self.codepath+spreadsheetname+".csv",index=False,mode='a+')
        filename_list = []
        for n in filenames:
            filename_list.append(re.search('(.+?)_', n).group(1))
        preds_list = np.squeeze(preds).tolist()
        file_dict = {}
        names = set(filename_list)
        FPs_avg = []
        FNs_avg = []
        FPs_MV = []
        FNs_MV = []
        error = 0
        #corrects = []
        variances = []
        for name in names:
            tilelist = [preds_list[i] for i in range(len(preds_list)) if filename_list[i] == name]
            yes = 0
            FN = 0
            FP = 0
            for tile in tilelist:
                if (self.artistname in name and not "Not" in name and tile < .5):
                    FN += 1
                    error += abs(tile - .5)
                elif ((self.artistname not in name or "Not" in name) and tile >= .5):
                    FP += 1
                    error += abs(tile - .5)
                else:
                    yes += 1
            if yes >= (FN+FP):
                maj_vote = "Y"
            else:
                maj_vote = "N"
                if FP>FN:
                    FPs_MV.append(1)
                else:
                    FNs_MV.append(1)
            MVfrac = yes/(yes+FN+FP)
            avg = np.mean(tilelist)
            var = np.var(tilelist)
            variances.append(var)            
            if (self.artistname in name and not "Not" in name and avg < .5):
                FNs_avg.append(1)
                file_dict[name] = (round(avg,6), round(var,6), "N", len(tilelist), maj_vote, MVfrac)
            elif ((self.artistname not in name or "Not" in name) and avg >= .5):
                FPs_avg.append(1)
                file_dict[name] = (round(avg,6), round(var,6), "N", len(tilelist), maj_vote, MVfrac)
            else:
                #corrects.append(1)
                file_dict[name] = (round(avg,6), round(var,6), "Y", len(tilelist), maj_vote, MVfrac)
        tot_incorrect_avg = sum(FPs_avg + FNs_avg)    
        tot_correct_avg = len(names) - tot_incorrect_avg
        tot_incorrect_MV = sum(FPs_MV + FNs_MV)
        tot_correct_MV = len(names) - tot_incorrect_MV
        if filewrite:
            f = open(self.codepath+spreadsheetname+".csv", "a+")
            f.write("\n\nModel: "+self.name)
            f.write("\n"+str(self.img_size)+"-pixel tiles\n")
            f.write("\nName,Average,Variance,Correct?,# tiles,Maj Vote,MV Frac Correct\n")
            for k in file_dict.keys():
                f.write(k+","+str(file_dict[k][0])+","+str(file_dict[k][1])+","+str(file_dict[k][2])+","+str(file_dict[k][3])+" tiles,"+file_dict[k][4]+","+str(file_dict[k][5])+"\n")
            f.write("\nAccuracy by Average: "+str(tot_correct_avg)+" / "+str(tot_incorrect_avg+tot_correct_avg)+" = "+str(round(1.*tot_correct_avg/(tot_correct_avg+tot_incorrect_avg+.00001),2)))
            f.write("\nAccuracy by Majority Vote: "+str(tot_correct_MV)+" / "+str(tot_incorrect_MV+tot_correct_MV)+" = "+str(round(1.*tot_correct_MV/(tot_correct_MV+tot_incorrect_MV+.00001),2)))            
            f.write("\nMean variance: "+str(1.*sum(variances)/(len(variances)+.0001)))
            f.write("\n\n"+str(sum(FPs_avg))+" False Positives by Average\n")
            f.write(str(sum(FNs_avg))+" False Negatives by Average\n")
            f.write("\n"+str(sum(FPs_MV))+" False Positives by Majority Vote\n")
            f.write(str(sum(FNs_MV))+" False Negatives by Majority Vote\n")
            f.write("\n\nTotal error = "+str(error)+"\n")
            f.close()
        return round(1.*tot_correct_avg/(tot_correct_avg+tot_incorrect_avg+.00001),2), \
            round(1.*tot_correct_MV/(tot_correct_MV+tot_incorrect_MV+.00001),2), \
            1.*sum(variances)/(len(variances)+.0001)
   
    def saveModel(self):
        self.model.save(self.codepath+self.name+'h5')

    def getName(self):
        return self.name

    def getModel(self):
        return self.model
        
    def getModelPath(self):
        return self.modelpath
    
    def getImageSize(self):
        return self.img_size
    
    def __str__(self):
        return 'Model name is '+self.name
    

class cnn_flex(object):
    def __init__(self, img_size, model_type, name, artistname, codepath, modelpath=None, new=False, batch_size=16, epochs=50, patience=5, save_every_epoch=False):
        self.img_size = img_size
        self.model_type = model_type
        self.codepath = codepath        
        self.modelpath = modelpath
        self.name = name
        self.artistname = artistname
        self.batch_size = batch_size
        self.new = new
        self.epochs = epochs
        self.patience = patience
        self.save_every_epoch = save_every_epoch
        if self.new:
            self.model = cnn_flex.makeModel(self.model_type, self.img_size, self.name)
        else:
            self.model = load_model(self.modelpath+self.name+'.h5')
            print("Model "+name+" loaded")
            
    def makeModel(model_type, img_size, name):
        if model_type == 'vgg16':
            newmodel = applications.VGG16(weights='imagenet', include_top=False, input_shape = (img_size,img_size,3))
        elif model_type == 'resnet50':
            newmodel = applications.ResNet50(weights='imagenet', include_top=False, input_shape = (img_size,img_size,3))
        else:
            print("Specify a valid model type")
            return
        model = Sequential()
        model.add(newmodel)
        
        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dense(img_size, activation='relu'))  #a hyperparameter
        top_model.add(Dropout(0.2))
        top_model.add(Dense(int(img_size/2), activation='relu'))  #a hyperparameter
        top_model.add(Dropout(0.3))
        top_model.add(Dense(1, activation = 'sigmoid'))
        
        model.add(top_model)
        
        print(model_type+" model "+name+" created or instantiated.\n")
        return model
    
    def train(self, trainpath, validatepath, l_rate=.001):
        model = self.model
        nb_train_samples = sum(len(files) for _, _, files in os.walk(trainpath))
        nb_validation_samples = sum(len(files) for _, _, files in os.walk(validatepath))
        epochs = self.epochs
        batch_size = self.batch_size
        if self.save_every_epoch:            
            mc = ModelCheckpoint(filepath=self.codepath+"saved-3Dmodel-"+str(self.img_size)+"-{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}.h5", \
                    monitor='val_acc', verbose=1, save_best_only=False, mode='max')
            callbacks_list = [mc]
        else:
            es = EarlyStopping(monitor='val_acc', mode='min', verbose=1, patience=self.patience) 
            mc = ModelCheckpoint(filepath=self.codepath+"best_3D-"+self.name+".h5", \
                    monitor='val_acc', verbose=1, save_best_only=True)
            callbacks_list = [es, mc]
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=l_rate),
                      metrics=['accuracy'])
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                            horizontal_flip=False)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(
                trainpath,
                target_size=(self.img_size, self.img_size),
                batch_size=self.batch_size,
                class_mode='binary')
        validation_generator = test_datagen.flow_from_directory(
                validatepath,
                target_size=(self.img_size, self.img_size),
                batch_size=batch_size,
                class_mode='binary')
        print("Training "+self.name+" with tiles in "+trainpath)
        model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples // batch_size,
                epochs=epochs,
                callbacks=callbacks_list,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples // batch_size)
        print(self.name+" trained successfully.\n")
        return
    
    def test(self, testpath, spreadsheetname, filewrite=True):
        model = self.model
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
                directory=testpath,
                target_size=(self.img_size, self.img_size),
                color_mode="rgb",
                batch_size=1,
                class_mode='binary',
                shuffle=False,    
                )
        print("Testing "+self.name+" with tiles in "+testpath)
        filenames = test_generator.filenames
        nb_samples = len(filenames)
        test_generator.reset()
        preds = model.predict_generator(test_generator,nb_samples)
        filename_list = []
        for n in filenames:
            filename_list.append(re.search('(?<=/)(.+?)_', n).group(1))
        preds_list = np.squeeze(preds).tolist()
        file_dict = {}
        names = set(filename_list)
        FPs_avg = []
        FNs_avg = []
        FPs_MV = []
        FNs_MV = []
        error = 0
        #corrects = []
        variances = []
        for name in names:
            tilelist = [preds_list[i] for i in range(len(preds_list)) if filename_list[i] == name]
            yes = 0
            FN = 0
            FP = 0
            for tile in tilelist:
                if (self.artistname in name and not "Not" in name and tile < .5):
                    FN += 1
                    error += abs(tile - .5)
                elif ((self.artistname not in name or "Not" in name) and tile >= .5):
                    FP += 1
                    error += abs(tile - .5)
                else:
                    yes += 1
            if yes >= (FN+FP):
                maj_vote = "Y"
            else:
                maj_vote = "N"
                if FP>FN:
                    FPs_MV.append(1)
                else:
                    FNs_MV.append(1)
            MVfrac = yes/(yes+FN+FP)
            avg = np.mean(tilelist)
            var = np.var(tilelist)
            variances.append(var)            
            if (self.artistname in name and not "Not" in name and avg < .5):
                FNs_avg.append(1)
                file_dict[name] = (round(avg,6), round(var,6), "N", len(tilelist), maj_vote, MVfrac)
            elif ((self.artistname not in name or "Not" in name) and avg >= .5):
                FPs_avg.append(1)
                file_dict[name] = (round(avg,6), round(var,6), "N", len(tilelist), maj_vote, MVfrac)
            else:
                #corrects.append(1)
                file_dict[name] = (round(avg,6), round(var,6), "Y", len(tilelist), maj_vote, MVfrac)
        tot_incorrect_avg = sum(FPs_avg + FNs_avg)    
        tot_correct_avg = len(names) - tot_incorrect_avg
        tot_incorrect_MV = sum(FPs_MV + FNs_MV)
        tot_correct_MV = len(names) - tot_incorrect_MV
        if filewrite:
            f = open(self.codepath+spreadsheetname+".csv", "a+")
            f.write("\n\nModel: "+self.name)
            f.write("\n"+str(self.img_size)+"-pixel tiles\n")
            f.write("\nName,Average,Variance,Correct?,# tiles,Maj Vote,MV Frac Correct\n")
            for k in file_dict.keys():
                f.write(k+","+str(file_dict[k][0])+","+str(file_dict[k][1])+","+str(file_dict[k][2])+","+str(file_dict[k][3])+" tiles,"+file_dict[k][4]+","+str(file_dict[k][5])+"\n")
            f.write("\nAccuracy by Average: "+str(tot_correct_avg)+" / "+str(tot_incorrect_avg+tot_correct_avg)+" = "+str(round(1.*tot_correct_avg/(tot_correct_avg+tot_incorrect_avg+.00001),2)))
            f.write("\nAccuracy by Majority Vote: "+str(tot_correct_MV)+" / "+str(tot_incorrect_MV+tot_correct_MV)+" = "+str(round(1.*tot_correct_MV/(tot_correct_MV+tot_incorrect_MV+.00001),2)))            
            f.write("\nMean variance: "+str(1.*sum(variances)/(len(variances)+.0001)))
            f.write("\n\n"+str(sum(FPs_avg))+" False Positives by Average\n")
            f.write(str(sum(FNs_avg))+" False Negatives by Average\n")
            f.write("\n"+str(sum(FPs_MV))+" False Positives by Majority Vote\n")
            f.write(str(sum(FNs_MV))+" False Negatives by Majority Vote\n")
            f.write("\n\nTotal error = "+str(error)+"\n")
            f.close()
        return round(1.*tot_correct_avg/(tot_correct_avg+tot_incorrect_avg+.00001),2), \
            round(1.*tot_correct_MV/(tot_correct_MV+tot_incorrect_MV+.00001),2), \
            1.*sum(variances)/(len(variances)+.0001)    
    
    def saveModel(self):
        self.model.save(self.codepath+self.name+'h5')

    def getName(self):
        return self.name

    def getModel(self):
        return self.model
        
    def getModelPath(self):
        return self.modelpath
    
    def getImageSize(self):
        return self.img_size
    
    def __str__(self):
        return 'Model name is '+self.name