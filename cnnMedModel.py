#Copyright (c) 2019-2020 Steven J. Frank
#All rights reserved.


import numpy as np
import pandas as pd
import os
import re
import glob
import imageio
import skimage
import cv2
from keras import applications
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, BatchNormalization
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from tensorflow.keras.utils import Sequence
from keras.utils import np_utils
from keras import backend as K
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.layers import Input
from keras.preprocessing import image
from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
from skimage import data, color, io, img_as_float


class cnnMed5_3D(object):
    def __init__(self, img_size, name, type1, type2, codepath, modelpath=None, new=False, batch_size=16, epochs=50, patience=50, save_every_epoch=False):
        self.img_size = img_size
        self.codepath = codepath        
        self.modelpath = modelpath
        self.name = name
        self.type1 = type1
        self.type2 = type2
        self.batch_size = batch_size
        self.new = new
        self.epochs = epochs
        self.patience = patience
        self.save_every_epoch = save_every_epoch
        if self.new:
            self.model = cnnMed5_3D.makeModel(self.img_size, self.name)
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
    
    def train(self, trainpath, validatepath, l_rate=.001):
        os.makedirs(trainpath, exist_ok=True)
        model = self.model
        nb_train_samples = sum(len(files) for _, _, files in os.walk(trainpath))
        nb_validation_samples = sum(len(files) for _, _, files in os.walk(validatepath))
        epochs = self.epochs
        batch_size = self.batch_size
        if self.save_every_epoch:            
            mc = ModelCheckpoint(filepath=self.codepath+"saved-cnnMed5_3D-model-"+str(self.img_size)+"-{epoch:02d}-{val_loss:.2f}.h5", \
                    monitor='val_loss', verbose=1, save_best_only=False, mode='max')
            callbacks_list = [mc]
        else:
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.patience) 
            mc = ModelCheckpoint(filepath=self.codepath+"best_cnnMed5_3D-model"+self.name+".h5", \
                    monitor='val_loss', verbose=1, save_best_only=True)
            callbacks_list = [es, mc]
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=l_rate),
                      metrics=['accuracy'])
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                            horizontal_flip=True, vertical_flip=True)
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
            #filename_list.append(re.search('(?<=/)(.+?)_', n).group(1))
            #n1 = re.search('[ \w-]+?(?=\.)', n).group(0)  #for local computer
            n1 = re.search("^.*?(?=_)", n).group(0)  #for tilenames with coords
            filename_list.append(re.search(r'[^_]*',n1).group(0))
        preds_list = np.squeeze(preds).tolist()
        file_dict = {}
        names = set(filename_list)
        FT2s_avg = []
        FT1s_avg = []
        FT2s_MV = []
        FT1s_MV = []
        error = 0
        variances = []
        MVfrac_list = []
        for name in names:
            tilelist = [preds_list[i] for i in range(len(preds_list)) if filename_list[i] == name]
            yes = 0
            FT1 = 0
            FT2 = 0
            error_list = []
            for tile in tilelist:
                #if (self.type1 in name and tile > .5):
                if (self.type2 in name and tile < .5):
                    #FT1 += 1
                    FT2 += 1
                    error_list.append(abs(tile - .5))
                #elif (self.type1 not in name and tile <= .5):
                elif (self.type2 not in name and tile >= .5):
                    FT1 += 1
                    error_list.append(abs(tile - .5))
                else:
                    yes += 1
            if yes >= (FT1+FT2):
                maj_vote = "Y"
            else:
                maj_vote = "N"
                if FT2>FT1:
                    FT2s_MV.append(1)
                else:
                    FT1s_MV.append(1)
            if len(error_list) > 0:
                error += sum(error_list)/len(error_list)
            MVfrac = yes/(yes+FT1+FT2)
            MVfrac_list.append(MVfrac)
            avg = np.mean(tilelist)
            var = np.var(tilelist)
            variances.append(var)            
            #if (self.type1 in name and avg > .5):
            if (self.type2 in name and tile < .5):
                # FT1s_avg.append(1)
                # file_dict[name] = (round(avg,6), round(var,6), "N", len(tilelist), maj_vote, MVfrac)
                FT2s_avg.append(1)
                file_dict[name] = (round(avg,6), round(var,6), "N", len(tilelist), maj_vote, MVfrac)

            #elif (self.type1 not in name and avg <= .5):
            elif (self.type2 not in name and tile >= .5):
                # FT2s_avg.append(1)
                # file_dict[name] = (round(avg,6), round(var,6), "N", len(tilelist), maj_vote, MVfrac)
                FT1s_avg.append(1)
                file_dict[name] = (round(avg,6), round(var,6), "N", len(tilelist), maj_vote, MVfrac)
            else:
                file_dict[name] = (round(avg,6), round(var,6), "Y", len(tilelist), maj_vote, MVfrac)
        tot_incorrect_avg = sum(FT2s_avg + FT1s_avg)    
        tot_correct_avg = len(names) - tot_incorrect_avg
        tot_incorrect_MV = sum(FT2s_MV + FT1s_MV)
        tot_correct_MV = len(names) - tot_incorrect_MV
        frac_correct = sum(MVfrac_list)/len(MVfrac_list)
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
            f.write("\n\n"+str(sum(FT2s_avg))+" False "+self.type1+" classifications by Average\n")
            f.write(str(sum(FT1s_avg))+" False "+self.type2+" classifications by Average\n")
            f.write("\n"+str(sum(FT2s_MV))+" False "+self.type1+" classifications by Majority Vote\n")
            f.write(str(sum(FT1s_MV))+" False "+self.type2+" classifications by Majority Vote\n")
            f.write("\n\nTotal error = "+str(error))
            f.write("\n\nTile fraction correctly classified = "+str(round(frac_correct,2))+"\n")
            f.close()
        return round(1.*tot_correct_avg/(tot_correct_avg+tot_incorrect_avg+.00001),2), \
            round(1.*tot_correct_MV/(tot_correct_MV+tot_incorrect_MV+.00001),2), \
            1.*sum(variances)/(len(variances)+.0001), round(frac_correct,2)
            
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
    
    def loadWeights(self, weight_file):
        self.model.load_weights(weight_file)
    
    def __str__(self):
        return 'Model name is '+self.name
    

class cnnMed5_3D_3way(object):    
    def __init__(self, img_size, name, type1, type2, codepath, modelpath=None, new=False, batch_size=16, epochs=40, patience=40, save_every_epoch=False, weights=None):
        self.img_size = img_size
        self.codepath = codepath        
        self.modelpath = modelpath        
        self.name = name
        self.type1 = type1
        self.type2 = type2
        self.batch_size = batch_size
        self.new = new
        self.epochs = epochs
        self.patience = patience
        self.save_every_epoch = save_every_epoch
        if self.new:
            self.model = cnnMed5_3D_3way.makeModel(self.img_size, self.name)
            if modelpath:
                os.makedirs(modelpath, exist_ok=True)
            if weights:
                self.model.load_weights(weights)
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
        model.add(Dense(3, activation='softmax'))
        print("3-way model "+name+" created or instantiated.\n")        
        return model
    
    def train(self, trainpath, validatepath, l_rate=.001, savepath=None):
        model = self.model
        nb_train_samples = sum(len(files) for _, _, files in os.walk(trainpath))
        nb_validation_samples = sum(len(files) for _, _, files in os.walk(validatepath))
        epochs = self.epochs
        batch_size = self.batch_size
        if not savepath:
            savepath = self.modelpath
        if self.save_every_epoch:            
            mc = ModelCheckpoint(filepath=savepath+"saved-cnnMed5_3D-3way-model-"+str(self.img_size)+"-{epoch:02d}-{val_loss:.2f}.h5", \
                    monitor='val_loss', verbose=1, save_best_only=False, mode='max')
            callbacks_list = [mc]
        else:
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.patience) 
            mc = ModelCheckpoint(filepath=savepath+"best_cnnMed5_3D-3way-model"+self.name+".h5", \
                    monitor='val_loss', verbose=1, save_best_only=True)
            callbacks_list = [es, mc]
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=l_rate),
                      metrics=['accuracy'])
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                            horizontal_flip=True, vertical_flip=True)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(
                trainpath,
                target_size=(self.img_size, self.img_size),
                batch_size=self.batch_size,
                class_mode='categorical')
        validation_generator = test_datagen.flow_from_directory(
                validatepath,
                target_size=(self.img_size, self.img_size),
                batch_size=batch_size,
                class_mode='categorical')
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
    
    def testAndMap(self, imagename, imagepath, basetilepath, maskpath, smooth=False, savemask=False, savemap=True, mapname=None, labeled=True, st_el_size=100):
        model = self.model
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
                directory=basetilepath+"test"+str(self.img_size)+"/",
                target_size=(self.img_size, self.img_size),
                color_mode="rgb",
                batch_size=1,
                class_mode='categorical',
                shuffle=False,    
                )        
        #test_generator = test_datagen.flow(imlist, [], [], batch_size=1)
        print("Testing "+self.name+" with "+imagename+" tiles")
        filenames = test_generator.filenames
        nb_samples = len(filenames)
        test_generator.reset()
        preds = model.predict_generator(test_generator,nb_samples)        
        predicted_class_indices=np.argmax(preds,axis=1)
        labels = (test_generator.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        predictions = [labels[k] for k in predicted_class_indices]
        image_predictions_list = []
        for i in range(len(predictions)):
            if imagename in filenames[i]:
                image_predictions_list.append(predictions[i])
        print("Found "+str(len(image_predictions_list))+" "+imagename+" tiles")
        type1_tot = sum([1 for p in image_predictions_list if p == self.type1])
        type2_tot = sum([1 for p in image_predictions_list if p == self.type2])
        dom = self.type1 if type1_tot > type2_tot else self.type2
        print("Dominant type for "+imagename+" is "+dom+" based on "+str(type1_tot)+" type 1 ("+self.type1+") predictions and "+str(type2_tot)+" type 2 ("+self.type2+") predictions")
        correct=None
        if labeled:
            if dom in imagename:
                print(imagename+" classified correctly")
                correct = True
            else:
                print("Incorrect classification for "+imagename)
                correct = False
        tilelist = [name for name in filenames if (predictions[filenames.index(name)] == dom and imagename in name)]
        print("Mapping with "+str(len(tilelist))+" tiles") 
        if not mapname:
            mapname = self.name+" size "+str(self.img_size)+" tiles of "+imagename+" with prediction "+dom
        rgb_img = Image.open(imagepath+imagename+'.jpg')
        grayscale = rgb_img.convert('L')
        grayarray = np.array(grayscale)
        rows, cols = grayarray.shape
        color_mask = Image.fromarray(np.zeros((rows, cols, 3), dtype=np.uint8))
        for t in tilelist:
            name = os.path.basename(t)
            coords = [int(x) for x in re.findall('\\((.*?)\\)', name)[0].split(',')] #get coordinate tuple
            draw = ImageDraw.Draw(color_mask)
            draw.rectangle(coords, fill="red")
            del draw        
        if smooth:
            cv2mask = np.array(color_mask)            
            cv2colormask = cv2.cvtColor(cv2mask, cv2.COLOR_RGB2BGR) #convert to cv2
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (st_el_size, st_el_size))
            cv2_color_mask = cv2.morphologyEx(cv2colormask, cv2.MORPH_OPEN, kernel, iterations=3)
            color_mask = Image.fromarray(cv2_color_mask)
            if savemask:
                cv2bwmask = cv2.cvtColor(cv2mask, cv2.COLOR_RGB2GRAY) #convert to cv2
                cv2_bw_mask = cv2.morphologyEx(cv2bwmask, cv2.MORPH_OPEN, kernel, iterations=3)
                pil_bw_mask = Image.fromarray(cv2_bw_mask)
                pil_bw_mask = pil_bw_mask.point(lambda x: 0 if x<1 else 255, '1')
                pil_bw_mask.save(maskpath+self.name+'_'+imagename+'.jpg')
                savemask=False
        if savemask: 
            cv2mask = np.array(color_mask)            
            cv2bwmask = cv2.cvtColor(cv2mask, cv2.COLOR_RGB2GRAY)            
            bw_mask = Image.fromarray(cv2bwmask)
            bw_mask = bw_mask.point(lambda x: 0 if x<1 else 255, '1')
            bw_mask.save(maskpath+self.name+'_'+imagename+'_mask.jpg')
        alpha = 0.75
        img_color = np.dstack((grayscale, grayscale, grayscale))
        # Convert the input image and color mask to Hue Saturation Value (HSV)
        # colorspace
        img_hsv = color.rgb2hsv(img_color)
        color_mask_hsv = color.rgb2hsv(color_mask)
        # Replace the hue and saturation of the original image
        # with that of the color mask        
        img_hsv[..., 0] = color_mask_hsv[..., 0]
        img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha
        img_masked = color.hsv2rgb(img_hsv)
        # Use keras to convert, and save
        img_masked = image.array_to_img(img_masked)
        if savemap:
            img_masked.save(self.codepath+mapname+'.jpg')        
        return dom, correct
 
    def classifyTiles(self, basetilepath, spreadsheetname):
        model = self.model
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
                directory=basetilepath+"test"+str(self.img_size)+"/",
                target_size=(self.img_size, self.img_size),
                color_mode="rgb",
                batch_size=1,
                class_mode='categorical',
                shuffle=False,    
                )        
        #test_generator = test_datagen.flow(imlist, [], [], batch_size=1)
        print("Testing "+self.name+" with tiles from "+basetilepath)
        filenames = test_generator.filenames
        nb_samples = len(filenames)
        test_generator.reset()
        preds = model.predict_generator(test_generator,nb_samples)        
        predicted_class_indices=np.argmax(preds,axis=1)
        labels = (test_generator.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        predictions = [labels[k] for k in predicted_class_indices]
        filename_list = []
        for n in filenames:
            n1 = os.path.splitext(os.path.basename(n))[0]
            filename_list.append(n1[:n1.index('_tile')])
        names = set(filename_list)
        f = open(self.codepath+spreadsheetname+".csv", "a+")
        f.write("\n\nModel: "+self.name)
        f.write("\n"+str(self.img_size)+"-pixel tiles\n")
        f.write("\nName,# tiles "+self.type1+",# tiles "+self.type2+",Classification\n\n")
        f.close()
        for name in names:
            predslist = [predictions[i] for i in range(len(predictions)) if filename_list[i] == name]
            #print("Found "+str(len(predslist))+" predictions for "+name)
            type1_tot = sum([1 for p in predslist if p == self.type1])
            type2_tot = sum([1 for p in predslist if p == self.type2])
            dom = self.type1 if type1_tot > type2_tot else self.type2
            f = open(self.codepath+spreadsheetname+".csv", "a+")
            f.write(name+","+str(type1_tot)+","+str(type2_tot)+","+dom+"\n")
            f.close()
        return
 
    def classifyAndScore(self, imagepath, basetilepath, spreadsheetname):
        model = self.model
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
                directory=basetilepath+"test"+str(self.img_size)+"/",
                target_size=(self.img_size, self.img_size),
                color_mode="rgb",
                batch_size=1,
                class_mode='categorical',
                shuffle=False,    
                )        
        #test_generator = test_datagen.flow(imlist, [], [], batch_size=1)        
        filenames = test_generator.filenames
        nb_samples = len(filenames)
        test_generator.reset()
        preds = model.predict_generator(test_generator,nb_samples)        
        predicted_class_indices=np.argmax(preds,axis=1)
        labels = (test_generator.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        predictions = [labels[k] for k in predicted_class_indices]        
        imagelist = glob.glob(imagepath+"*.jpg")
        with open(self.codepath+spreadsheetname+".csv", "a+") as f:
            f.write("\n\nModel: "+self.name)
            f.write("\n"+str(self.img_size)+"-pixel tiles\n")
            f.write("\nImage Name,# tiles "+self.type1+",# tiles "+self.type2+",Classification,Correct?\n\n")
            FT1 = 0 #false type 1s
            FT2 = 0 #false type 2s
            correct = 0
            for filename in imagelist:
                imagename = os.path.splitext(os.path.basename(filename))[0]
                print("Testing "+self.name+" with "+imagename+" tiles")
                image_predictions_list = []
                for i in range(len(predictions)):
                    if imagename in filenames[i]:
                        image_predictions_list.append(predictions[i])
                type1_tot = sum([1 for p in image_predictions_list if p == self.type1])
                type2_tot = sum([1 for p in image_predictions_list if p == self.type2])
                dom = self.type1 if type1_tot > type2_tot else self.type2
                if dom in imagename and not (type1_tot == 0 and type2_tot == 0):
                    f.write(imagename+","+str(type1_tot)+","+str(type2_tot)+","+dom+",Yes\n")
                    correct += 1
                if not dom in imagename and not (type1_tot == 0 and type2_tot == 0):
                    if dom == self.type1:                                
                        FT1 += 1
                    else:
                        FT2 += 1
            acc = correct / (correct + FT1 + FT2 + .01) #prevent division by zero
            f.write("\nAccuracy:  "+str(acc)+"\n")
            f.write(str(FT1)+" False "+self.type1+" classifications\n")
            f.write(str(FT2)+" False "+self.type2+" classifications\n")
            
        
    def classifyAndScorefromCSV(self, csvpath, csvname, basetilepath, spreadsheetname):
        model = self.model
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
                directory=basetilepath+"test"+str(self.img_size)+"/",
                target_size=(self.img_size, self.img_size),
                color_mode="rgb",
                batch_size=1,
                class_mode='categorical',
                shuffle=False,    
                )        
        #test_generator = test_datagen.flow(imlist, [], [], batch_size=1)        
        filenames = test_generator.filenames
        nb_samples = len(filenames)
        test_generator.reset()
        preds = model.predict_generator(test_generator,nb_samples)        
        predicted_class_indices=np.argmax(preds,axis=1)
        labels = (test_generator.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        predictions = [labels[k] for k in predicted_class_indices]        
        #imagelist = glob.glob(imagepath+"*.jpg")
        with open(csvpath+csvname+".csv",'r') as f:
            imagelist = []
            for line in f:
                data_line = line.rstrip().split(',')
                if not ('train' in data_line[0] or 'test' in data_line[0]):
                    imagelist.append(data_line[0])   
            while('' in imagelist): 
                imagelist.remove('')
        with open(self.codepath+spreadsheetname+".csv", "a+") as f:
            f.write("\n\nModel: "+self.name)
            f.write("\n"+str(self.img_size)+"-pixel tiles\n")
            f.write("\nImage Name,# tiles "+self.type1+",# tiles "+self.type2+",Classification,Correct?\n\n")
            FT1 = 0 #false type 1s
            FT2 = 0 #false type 2s
            correct = 0
            for filename in imagelist:
                #imagename = os.path.splitext(os.path.basename(filename))[0]
                print("Testing "+self.name+" with "+filename+" tiles")
                image_predictions_list = []
                for i in range(len(predictions)):
                    if filename in filenames[i]:
                        image_predictions_list.append(predictions[i])
                type1_tot = sum([1 for p in image_predictions_list if p == self.type1])
                type2_tot = sum([1 for p in image_predictions_list if p == self.type2])
                dom = self.type1 if type1_tot > type2_tot else self.type2
                if dom in filename and not (type1_tot == 0 and type2_tot == 0):
                    f.write(filename+","+str(type1_tot)+","+str(type2_tot)+","+dom+",Yes\n")
                    correct += 1
                if not dom in filename and not (type1_tot == 0 and type2_tot == 0):
                    f.write(filename+","+str(type1_tot)+","+str(type2_tot)+","+dom+",No\n")
                    if dom == self.type1:                                
                        FT1 += 1
                    else:
                        FT2 += 1
            acc = correct / (correct + FT1 + FT2 + .01) #prevent division by zero
            f.write("\nAccuracy:  "+str(acc)+"\n")
            f.write(str(FT1)+" False "+self.type1+" classifications\n")
            f.write(str(FT2)+" False "+self.type2+" classifications\n")


    def getPreds(self, imagename, basetilepath):
        model = self.model
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
                directory=basetilepath+"test"+str(self.img_size)+"/",
                target_size=(self.img_size, self.img_size),
                color_mode="rgb",
                batch_size=1,
                class_mode='categorical',
                shuffle=False,    
                )  
        print("Testing "+self.name+" with tiles in "+basetilepath)
        #filenames = [f for f in test_generator.filenames if imagename in f]
        filenames = test_generator.filenames
        nb_samples = len(filenames)
        test_generator.reset()
        preds = model.predict_generator(test_generator,nb_samples)        
        preds_list = np.squeeze(preds).tolist()
        return filenames, preds_list, (test_generator.class_indices)
    

    def saveModel(self):
        self.model.save(self.codepath+self.name+'.h5')

    def getName(self):
        return self.name

    def getModel(self):
        return self.model
        
    def getModelPath(self):
        return self.modelpath
    
    def getImageSize(self):
        return self.img_size
    
    def loadWeights(self, weight_path, weight_file):
        print("Loading weights from "+weight_path+weight_file+"\n")
        self.model.load_weights(weight_path+weight_file+'.h5')        
        
    def saveWeights(self, weight_path):
        self.model.save_weights(weight_path+self.name+"_weights.h5")
    
    def __str__(self):
        return 'Model name is '+self.name
    
