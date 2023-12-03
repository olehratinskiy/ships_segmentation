# ships_segmentation

###This project uses UNet model to segment ships on images.

####Its structure:
1) train_model.py - file with loading and preprocessing data, model creation and training.
2) test_model.py - file which loads model from file and gets prediction. Also uses dice score to check prediction accuracy.
3) ships-segmentation.ipynb - notebook with combined code from previous two files and some ntermediate steps.
4) requirements.txt - file with all libraries used in project

### To run and test this project you need:
1) clone this repo
2) install all requirements stated in requirements.txt
3) create data folder, place in it train_ship_segmentations_v2.csv and also create inside data folder train_v2 with all original stellite images 
4) run train_model.py and test_model.py