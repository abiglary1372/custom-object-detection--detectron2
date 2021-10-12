
''' introduction:
    
    author: Amin Biglary Makvand
    
    the goal of this script is to train an object detection model for cars
    then identifies the red cars
    then output a CSV containing the information on the inferences made
    for this task the following tools were used :
        an ubuntu-based system 
        anaconda package to create a virtual environment 
        detectron2 the Facebook computer vision library 
        CUDAtoolkit to enable GPU computation
        ****note: a *.yaml file is provided for the user to be able to replicate the 
        virtual env with all the dependencies exactly 
        
    here is a discribtion of the project directory:
        objectDetection/
               main.py
               data/
                   output/
                   rawDataset/
                   test/
                   train/
                   valid/
                   customTestDataset/
                    
               trainedModel/
                   eval/
                   savedModles/
                   
          explaining each directory:
              1- main.py is the main script for running the entire pipeline
              2-"output" is the directory where the final CSV is stored
              3- "rawDataset" is where we put the provided dataset (images+json file)
              4- "test" is where the test data set is created after the function prepare_data_sets() is executed 
              5- "train" is where the training data set is created after the function prepare_data_sets() is executed 
              6- "valid" is where the validation data set is created after the function prepare_data_sets() is executed 
              7- traineModel is where the final model is stored after completing the training job
              it is important to note that if you already have a trained model and don't want to run the training job again
              copy your model here and comment out the training and evaluation lines in the main
              8- savedModels is where you can save your trained models
              9- eval is where evaluation data is saved after running the evaluation job
              10- customTestDataset/ a custom test created by roboflow for further testing
        
        **commenting on the performance of the classification:
            the object detection model works well in detection of cars although it is observed that sometimes it considers
            trucks and motorcycles as cars too
            the color detection algorithm seems to be performing poorly specially for lower quality images and maybe a more
            complex algorithms are needed
            
            
'''
import detectron2
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer

# import some common libraries
import json
import shutil
import os
import pandas as pd
import numpy as np
import cv2
import random
import csv
import pathlib
import glob
import random


cfg = get_cfg()
mainPath = os.path.abspath(os.path.realpath(__file__))
mainPath = mainPath[:len(mainPath) - 7]

def main():
    '''  comments:   
    
        here the entire pipeline from data preprocessing to the final output is executed
        it is important to note that a trained model is already provided therefore the lines 139 140
        related model training and validation can be commented out, if u want to start a training job u have 
        to uncomment them.
        also, the line about visualizing the training data set is commented out. it visualizes the registered data
        set with annotations to make sure that the dataset is registered properly
        the line for running inference on the test dataset the function accepts a boolean which 
        when its false means that it won't visualize the model's output and true means otherwise
        
        below describes the entire project directory :
           objectDetection/
               main.py
               data/
                   output/
                   rawDataset/
                   test/
                   train/
                   valid/
                    
               trainedModel/
                   eval/
                   savedModles/
                   
          explaining each directory:
              1- main.py is the main script for running the entire pipeline
              2-"output" is the directory where the final CSV is stored
              3- "rawDataset" is where we put the provided dataset (images+json file)
              4- "test" is where the test data set is created after the function prepare_data_sets() is executed 
              5- "train" is where the training data set is created after the function prepare_data_sets() is executed 
              6- "valid" is where the validation data set is created after the function prepare_data_sets() is executed 
              7- traineModel is where the final model is stored after completing the training job
              it's important to note that if you already have a trained model and don't want to run the training job again
              copy your model here and comment out the training and evaluation lines in the main
              8- savedModels is where you can save your trained models
              9- eval is where evaluation data is saved after running the evaluation job
              
    '''
    cocoJson = os.path.join(mainPath,'data','rawDataSet','annotations_sample.json' );
    images = os.path.join(mainPath,'data','rawDataSet')
    show_red_car = False
    show_test_predictions = False
    
    prepare_data_sets(cocoJson,images)
    register_datasets()
    visualize_training_dataset()
    configure_model_for_training()
    trainer= run_training()
    run_evaluation(trainer)
    (boxPerIm,imname)=run_inference_on_test_set(show_test_predictions)
    (nameList, coordinateList, colorRed)=get_color(boxPerIm,imname,show_red_car)
    generate_output(nameList, coordinateList, colorRed)
    
    
def prepare_data_sets(filedir,directory):
    """ comments:
         this function prepares the provided data set to be used by detectron2
         the .json file that provides us with annotation data on our images is not 
         complete and lacks two keys "category" and "images" which the models need to start 
         the training and evaluation, therefore this function generates the proper coco dataset JSON file based on the 
         classes that we want to detect and the image files information.
         also, we need to divide the data set into three sets train validation and test. this function after generating the 
         coco JSON file divides the data set into those three sets and stores the values under this directory: .............
         so briefly the function does the following:
         1-first it extracts the annotation data from the provided JSON file
         2- it filters out any object annotations that are not cars
         3- it divides them into three sets training validation and test 
         4- it then copies the corresponding images of each set to their corresponding folder in the projects 
         directory
         5- then the two other necessary keys "category" and "images" are created and with the annotation data 
         are combined to gather to generate the JSON data structure for coco datasets for each of the three sets 
         6- then for each set the JSON file is saved in their corresponding directory 
	 """
    
    imList=[];
    
    file = open(filedir)
     
    #annotation data that is provided 
    annData = json.load(file);
                      
    annListFlt= annData["annotations"]      
    annDataflt = {"annotations": annListFlt}
    
    ##### dividing to validation test and train 
     
    length = len(annListFlt)
     
    # determining the traing validation and test percentages
    pt=0.7
    pv=0.2
     
    tre = int(length*pt); #end of train
    vs = tre; #start of vallidation
    ve = vs+ int(length*pv);# end of validation
    ts= ve; #start of test
    te = length; #end of test 
     
    imageTrain=[]
    imageValid=[]
    imageTest=[]
     
    trainSet=annData["images"][0:tre];
    validSet=annData["images"][vs:ve];
    testSet=annData["images"][ts:te];
    
    trainSetann=annData["annotations"][0:tre];
    validSetann=annData["annotations"][vs:ve];
    testSetann=annData["annotations"][ts:te];
     
    ###copying test files to test folder
    path=[];
    for diction in testSet:
        path.append(os.path.join(directory,str(diction["file_name"])))
    for f in path:
        shutil.copy(f, os.path.join(mainPath,'data','test'))
     
    ################copying the files for training
    trainPath=[];
    for diction in trainSet:
        trainPath.append(os.path.join(directory,str(diction["file_name"])))
    for f in trainPath:
        shutil.copy(f, os.path.join(mainPath,'data','train'))
     
    ###############copying the files for validation
    validPath=[];
    for diction in validSet:
        validPath.append(os.path.join(directory,str(diction["file_name"])))
    for f in validPath:
        shutil.copy(f, os.path.join(mainPath,'data','valid')) 
     
    #bulding the json files for each set 
    CocoFileTrain = {"categories": annData["categories"] , "images":trainSet ,"annotations": trainSetann };
    CocoFileValid = {"categories": annData["categories"] , "images":validSet ,"annotations": validSetann };
    CocoFileTest = {"categories": annData["categories"] , "images":testSet ,"annotations": testSetann };
 
 
#########saving the json files
with open(os.path.join(mainPath,'data','train','CocoAnnTrain.json'), 'w') as fp1:
    json.dump(CocoFileTrain, fp1)
with open(os.path.join(mainPath,'data','valid','CocoAnnValid.json'), 'w') as fp2:
    json.dump(CocoFileValid, fp2)
with open(os.path.join(mainPath,'data','test','CocoAnnTest.json'), 'w') as fp3:
    json.dump(CocoFileTest, fp3)
    
    
    #########saving the json files
    with open(os.path.join(mainPath,'data','train','CocoAnnTrain.json'), 'w') as fp1:
        json.dump(CocoFileTrain, fp1)
    with open(os.path.join(mainPath,'data','valid','CocoAnnValid.json'), 'w') as fp2:
        json.dump(CocoFileValid, fp2)
    with open(os.path.join(mainPath,'data','test','CocoAnnTest.json'), 'w') as fp3:
        json.dump(CocoFileTest, fp3)

def register_datasets():
    register_coco_instances("my_dataset_train", {}, os.path.join(mainPath,'data','train','CocoAnnTrain.json'), os.path.join(mainPath,'data','train'))
    register_coco_instances("my_dataset_val", {}, os.path.join(mainPath,'data','valid','CocoAnnValid.json'), os.path.join(mainPath,'data','valid'))
    register_coco_instances("my_dataset_test", {}, os.path.join(mainPath,'data','test','CocoAnnTest.json'), os.path.join(mainPath,'data','test'))

def visualize_training_dataset():
    my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
    dataset_dicts = DatasetCatalog.get("my_dataset_train")
    
    
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow('',vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
class CocoTrainer(DefaultTrainer):
  '''comments:
      Before starting training, we need to make sure that the model validates against our validation set. Unfortunately, this does not happen by default.
      We can easily do this by defining our custom trainer based on the Default Trainer with the COCO Evaluator. this is what we do with this class. 
      when the trainer is executed it will first call this class and does the validation and then the training starts
      
      '''
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        #os.makedirs("coco_eval", exist_ok=True)
        output_folder = os.path.join(mainPath,'trainedModel','coco_eval')

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

def configure_model_for_training():
    
    '''comments:
        here we configure our custom object detection model 
        we chose the FASTER RNN model from the model zoo
        the most important part here is the line .. where we configure whether we want CPU 
        computation for training or GPU if the goal is GPU the line must be commented out
        also for GPU the cfg.SOLVER.IMS_PER_BATCH  is very important especially when you get the runtime error 
    
    '''
    
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")) #***very important for inference
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.001
    
    
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 1400 #adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.STEPS = (1000, 1500)
    cfg.SOLVER.GAMMA = 0.05
    
    cfg.MODEL.DEVICE='cpu'
    
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 #your number of classes + 1 ***important for inference
    
    cfg.TEST.EVAL_PERIOD = 500
    cfg.OUTPUT_DIR = os.path.join(mainPath,"trainedModel")
    
def run_training():
    '''comments:
        here we first call the Cocotrainer class to validate the model against our validation dataset 
        then training job starts'''
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return trainer

def run_evaluation(trainer):
    '''comments:
    this function runs an inference on our test set and out puts a table of average Precisions AP which would
        give us an idea on how the model is performing'''
        
        
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir=os.path.join(mainPath,"trainedModel","eval"))
    val_loader = build_detection_test_loader(cfg, "my_dataset_test")
    inference_on_dataset(trainer.model, val_loader, evaluator)

def run_inference_on_test_set(vis=False):
    '''comments:
        in this function, we run inference on the trained model that we have stored in the directory
        this function is independent of the training and when we have a trained model we can use it to do inference on the test set
        this function also outputs three important data structures that are essential to the next part of the algorithm 
        which is detected the color red
        
        '''
    
    cfg.MODEL.WEIGHTS = os.path.join(mainPath,"trainedModel","model_final.pth")
    cfg.DATASETS.TEST = ("my_dataset_test", )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)
    test_metadata = MetadataCatalog.get("my_dataset_test")
    
    imCoordinates = [];
    boxPerIm = {}
    imname = []
    
    for imageName in glob.glob(os.path.join(mainPath,'data','test','*jpg')):
      im = cv2.imread(imageName)
      outputs = predictor(im)
      
      #saving arrays of box coordinates for each image in a list
      instance= outputs["instances"]._fields
      boxes= instance["pred_boxes"]
      boxestensor = boxes.tensor
      boxesArray = boxestensor.cpu().detach().numpy()
      imCoordinates.append(boxesArray)
      #########
      
      ###puting images and their coordinates in a dictionary 
      #boxPerIm[imageName].append(boxesArray)
      boxPerIm.setdefault(imageName,boxesArray)
      imname.append(imageName)

      
      #####################
      if vis:
          visualize_model_prediction(im,outputs,test_metadata)
          
    return boxPerIm,imname
          
def visualize_model_prediction(im,outputs,setMetadata):
    '''this function can be called to visualize the model output predictions '''
    v = Visualizer(im[:, :, ::-1],
                      metadata=setMetadata, 
                      scale=0.8
                        )
          
          
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('',out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def get_color(boxPerIm,imname,show):
    
    '''comments:
        
        this is the function that will run the algorithm we use to determine the color of a detected car 
        in the test set, here is how the algorithm works:
            1- the function recives the coordinates of the boundary boxes from the run_inference_on_test_set() function 
            2- then starts running through the test set directory loading images and cropping the cars out using the boundary box coordinates 
            3- then saves the cropped images as RGB and counts the number of pixels that have the RGB value of [190,x,x] to [255,x,x] which covers most shades of red
            4- then of these pixels make more than 10 percent of the total pixels in the image the car is detected 
            as red 
            
        commenting on the performance of the classification:
            the object detection model works well in detection of cars although it is observed that sometimes it considers
            trucks and motorcycles as cars too
            the color detection algorithm seems to be performing poorly specially for lower quality images and maybe a more
            complex algorithms are needed
        '''
    
    #[xmin,ymin,xmax,ymax] x width, y hight
    i=0;
    
    nameList =[]
    coordinateList=[]
    colorRed=[]
    nredList=[]
    for name in imname:
        coordinates = boxPerIm[name]
        i=0;
        for co in coordinates[:,0]:
            if i<coordinates.shape[0]:
                Wmin =int(coordinates[i][0])
                Hmin =int(coordinates[i][1])
                Wmax =int(coordinates[i][2])
                Hmax =int(coordinates[i][3])
                image = cv2.imread(name)
                image = image[Hmin:Hmax, Wmin:Wmax]
                size = image.shape
                imMask = np.zeros((size[0],size[1])) 
                imlayer = np.zeros((size[0],size[1])) 
                
                ####################
                #showing croping out the bunding box 
                
                # cv2.imshow("cropped", image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
                ####################
                
                ####estimating the cars color
                c1=0
                c2=0
                
                while c1<size[0]:
                    while c2<size[1]:
                        row=image[c1,c2,:]
                        row=row.reshape(-1,1)
                        
                        if 120<int(row[2]) and row[0]<60 and row[1]<60 and abs(row[1]-row[0])<30:
                            imMask[c1,c2] = 255
                        c2=c2+1
                    c2=0;
                    c1=c1+1
                    
                
                nRed = np.count_nonzero(imMask)
                nredList.append(nRed)
                ratio = nRed/(size[0]*size[1])
                
                nameList.append(os.path.basename(name))
                coordinateList.append(str([Wmin,Hmin,Wmax,Hmax]))
                
                if ratio>0.01:
                    colorRed.append(True)
                    if show:
                        show_red_car(image)
                        
                else:
                    colorRed.append(False)
                            
                #stacking to build the rgb mask#####
                stacked= np.stack((imlayer,imlayer,imMask), axis=2)
                # cv2.imshow('',stacked)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
                i=i+1;
    return nameList, coordinateList, colorRed
                
def generate_output(nameList, coordinateList, colorRed):
    
    '''comments:
        here a CSV file is created and then stored as the final output 
        the CSV file has three columns 
        the first column is the image file name
        second is the boundary box coordinates of the detected car 
        third is the true false column determining whether the car is red or not 
        
    '''
    finalOut={"file name" : nameList , "box coordinates": coordinateList , "is red": colorRed}; #the final csv out put 
    finalOutdf = pd.DataFrame.from_dict(finalOut)
    finalOutCSV = finalOutdf.to_csv(index=False)
    finalOutdf.to_csv(os.path.join(mainPath,'data','output','finalOutCSV.csv'), index = False)

def show_red_car(img):
    cv2.imshow("", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()

