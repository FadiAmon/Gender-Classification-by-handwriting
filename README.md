# Gender-Classification-by-handwriting
In this project, an SVM algorithm was created from a given dataset of handwritten images of Hebrew sentences to assign the images to their respective gender.

Authors:

1) Fadi Amon, EMAIL (contact info): fadiam@ac.sce.ac.il
2) Rasheed Abu Mdegem, EMAIL (contact info): rasheab1@ac.sce.ac.il

Description:

We made a python script in PyCharm environment which trains a SVM model on hand written images of text in Hebrew, the images were imported from HHD_gender data set.
The model was made using SVM classifier from sklearn with LBP feature extraction, the model recieves images of written text as np arrays,
and produces a prediction of the gender to each image based on the image features.
At the end, we produce a file called results.txt which shows how accuracies based on the model parameters, 
shows the results for test, and shows the confusion matrix.

Environment:

We used windows developing this script, all you need to run it is python and the needed libraries such as, OpenCV, sklearn,skimage,argparse, numpy and pandas.


Instructions on how to run the program:

In order to run the program you need to go to the terminal and type the name of the .py file and pass to it 3 arguments of the data set path (train, val and test).


Example:
*The data set in this exmaple is in the same directory as classifier.py, hence no full path was written.

python classifier.py ./HHD_gender/train ./HHD_gender/val ./HHD_gender/test   
