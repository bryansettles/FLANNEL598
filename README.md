# FLANNEL

## Data Prepare
### Data Collect
1. Download CCX data: from https://github.com/ieee8023/covid-chestxray-dataset, put them into original_data/covid-chestxray-dataset-master
2. Download KCX data from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia, put them into original_data/chest_xray

The data above has been added to the images folder already in this repository for running my code. Should be organized according to above instructions for instructions under Model Training

## Instructions for how to run my code:

In order to run this effectively, this software should be executed on a device with cuda.

cd FLANNEL/Code/

### Modify the second line of baselearn.py: 

modelType = 'resnet152' #set as needed

Modify the n_epochs of baselearn.py to modify the number of epochs


### To train the different models: 

#alexnet, shufflenet_v2_x1_0, resnet152, densenet161

### Then train base models by running the following for each model:

python baselearn.py

### Data Preprocessing, original experiment. From the Downloaded dataset of kaggle and covid-chestxray-dataset github repo, modify the paths so that the images are in inputImages and the metadata.csv is in a folder in FLANNEL/ titled 'original data'
1. extract data from CCX: data_preprocess/get_covid_data_dict.py 
2. extract data from KCX: data_preprocess/get_kaggle_data_dict.py
3. reorganize CCX&KCX data to generate 5-folder cross-validation expdata: data_preprocess/extract_exp_data_crossentropy.py

## Model Training, original experiment
### Base-modeler Learning
FLANNEL/ensemble_step1.py for 5 base-modeler learning [InceptionV3, Vgg19_bn, ResNeXt101, Resnet152, Densenet161]

(E.g. python ensemble_step1.py --arch InceptionV3)

### ensemble-model Learning
FLANNEL/ensemble_step2_ensemble_learning.py
