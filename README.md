# Project Description


The Heart disease is considered the major causes of the death worldwide, therefore early detection is necessary to prevent it.\
There is a significant potential for application of big data to healthcare,and the benefit of improving the patiant outcome, and minimise error in disease classification and diagnosis.\
In this project , we developed  machine learning  using  random forest classifier to classify and predict  heart disease. We used dataset from the UCI website [UCI](https://archive.ics.uci.edu/ml/index.ph).



## **Data source**
UCI - Machine Learning Repository;
Center for Machine Learning and Intelligent Systems
[The UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.ph) is a collection of databases, domain theories, and data generators that are used by the machine learning community for the empirical analysis of machine learning algorithms.\
[Blood Transfusion Service Center Data Set](https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center)

**Data**:
- [Download](https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/)

## **Installation**
```
## Import Libraries

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

%matplotlib inline
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import seaborn as sns
sns.set(style = 'ticks')
```


## **Data preparation**
The first step we load our dataset.\
then we applied the following techniques to get our data ready for our model.\
- Outliers (remove or adjust if possible or necessary)
- Null (remove or interpolate if possible or necessary)
- Missing Values (remove or interpolate if possible or necessary)
- Coded content (transform if possible or necessary [str to number or vice-versa])
- Normalisation (if possible or necessary)
 -Feature Engeneer (if useful or necessary)

## **Instantiate the model**

This invloving  spliting  the data into training and test 

## **Train(fit) the model**
Once we instaniate our model, we train and  fit our model, we consider its parameter and hyperparameter of each model to check the efficiency of the model against the training and test dataset.
We accommplished this by calling ```model.fit``` and passing the XY

## **Evaluate the model**
- We used the training data to make predictions, check for overfitting, and to determine the appropriate matrix for modelling.\
- We evaluate our model on (y,X)test, and printing its accuracy.\
- Since we are dealing with classification problem the  typical metrics for evaluating the model are Confusion matrix, accuracy, precision/recall, ROC

## The baseline results (minimum) are:
Accuracy = 0.7419
ROC AUC = 0.6150

## **Issues**
None

## **Contributing**
For major changes, please open an issue. First discuss what you would like to change. please make sure to update tests as appropriate

# **Licence**
```
Copyright (C) 2020 David Gabriel
```

# **Further reading & References**
-Yeh, I-Cheng, Yang, King-Jang, and Ting, Tao-Ming, "Knowledge discovery on RFM model using Bernoulli sequence, "Expert Systems with Applications, 2008\
- [The UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.ph)





