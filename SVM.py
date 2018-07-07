#!/usr/bin/env python

"""
    
Tutorial for alphabase.ai Competition (Basic Logistic Regression Model)


prerequisites: sklearn, pandas, numpy

install the required packages by 'pip install sklearn, pandas, numpy'

"""

import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from	sklearn	import	svm 

def main():
    ##################################
    ## Step 1. Load alphabase.ai data
    print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    print("Loading the data ...")
    # Load the data from the CSV files
    train = pd.read_csv('train.csv', header=0)  # Load the training data provided by alphabase.ai with Target.
    test = pd.read_csv('test.csv', header=0)    # Load the competition data provided by alphabase.ai with ID.
    
    ##
    #################################
    ##################################
    ## Step 2. Train the Logistic Regression Model
    # Prepare data, ignoring the NA-flag features
    #Y = train['Target']
    #print np.shape(train),train
    Y=np.array(train)[:,118]
    #print np.shape(Y),Y
    X_train = np.array(train)[:, :59]
    ID = test['ID']
    X_test = np.array(test)[:, 1:60]
    
    # Missing values imputation
    print("Missing values imputation ...")
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(X_train)
    X_train = imp.transform(X_train)
    X_test = imp.transform(X_test)
    
    ## Create a Logistic Regression Model.
	##clf=svm.SVC(kernel='linear') 
    print("Training the LR model ...")
    clf=svm.SVC() 
    #clf=svm.SVC(kernel='linear') 
    #clf=svm.SVC(kernel='poly',max_iter=200) 
    #clf=svm.SVC(kernel='poly')
    clf.probability=True
    clf.fit(X_train,Y)
    ##
    #################################
    ##################################
    ## Step 3. Predict the Competition Data with the newly trained model
    print("Predicting the Competition Data...")
   # y_test = model.predict_proba(X_test) # Predict the Target, getting the probability.
    #y_test=clf.predict(X_test)
    #kk=clf.decision_function(X_test)
    tt=clf.predict_proba(X_test)
    print np.shape(tt),tt
    #print(y_test);
    pred=tt[:, 1]                  # Get the probabilty of being 1.
    #pred_df = pd.DataFrame(data={'Target': pred})
    pred_df = pd.DataFrame(data={'Target': pred})
    submissions = pd.DataFrame(ID).join(pred_df)
    
    ##
    #################################
    ##################################
    ## Step 4. Write the CSV File and Get Ready for Submission
    # Save the predictions out to a CSV file
    print("Writing predictions to abai_submissions.csv")
    submissions.to_csv("svm_rbf_110000.csv", index=False)
    print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

    ##
    #################################





## Here the main program.

if __name__ == '__main__':
    main()


## Finish! You get your first model done! Upload to alphabase.ai and see your rank!
