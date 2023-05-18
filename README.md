# XGBoosting for Cancer Prediction

Machine learning techniques have become widely used for medical diagnosis. 
This model uses a simple XGBoost to predict whether tumors presented in the dataset are malignant or benign.
The [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download) was used with an 80/20 train-test split.
On the test set, the model predicted the nature of tumors 95.61% accurately with an F1 Score of 0.94.

The code was adapted from the [Deep Learning A-Z 2023](https://www.udemy.com/course/deeplearning/) course on Udemy.

## Running the project
From the root folder of the project, run 
`./install.sh`
to create the virtual environment and install the dependencies. Then, run `./run.sh` to execute the run file. 

