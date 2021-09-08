# custom object detection (detectron2)

## running the code on virual env

after installing all the dependacies using the *.yml file provided, creat the following project directory :

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
                   coco_eval
                   eval/
                   savedModles/
                   

This will be the directory where we are going to have our raw data as well as training validation and test set.

Following is the explanation for each directory folder:
explaining each directory:

* 1-"output" is the directory where the final CSV is stored
* 2- "rawDataset" is where we put the provided dataset (images+json file)
* 3- "test" is where the test data set is created after the function prepare_data_sets() is executed 
* 4- "train" is where the training data set is created after the function prepare_data_sets() is executed 
* 5- "valid" is where the validation data set is created after the function prepare_data_sets() is executed 
* 6- traineModel is where the final model is stored after completing the training job
it's important to note that if you already have a trained model and don't want to run the training job again
copy your model here and comment out the training and evaluation lines in the main
* 7- saved models is where you can save your trained models
* 8- eval is where evaluation data is saved after running the evaluation job
* 9- customTestDataset contains a dataset for further testing that i generated using roboflow

Therefore do the following: 

* upload the dataset into the "rawDataset" folder (it is alreedy there)
* if you don't want to train the model (it takes a lot of time) copy the model file from the "savedModles" folder (dont forget to comment out the training and validation lines "#trainer= run_training() #run_evaluation(trainer)" in the main function )
* run the script 

# Jupyter notebook 
if you don't want to go through the hassle of installing the dependencies just open the Jupyter notebook inside google collab 
