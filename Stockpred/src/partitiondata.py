# This module partitions data into training, test and forecast and returns the corresponding feature matrices and outputs
# Inputs: alldata - list containing the stock prices read from the csv file
#         modeldata  - number of data samples available for modeling
#         testpercent - percentage of data for testing.
#         lag - maximum number of lagged sample points to be included in the feature vector
#
# Outputs: Xtrain - training feature matrix, rows - number of features, columns - number of samples       
#          ytrain - output vector for training 
#          Xtest  - testing feature matrix         
#          ytest  - true output for test data
#          Xfcast - input matrix for forecasting future stock prices [Will be modified based on model analysis]

import numpy as np
from sklearn.cross_validation import train_test_split

def isFloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def extract_features(cur_index,cur_object,all_data,lag):
    dummy = [];
    [dummy.append(float(k)) for k in cur_object[1:]] # load current stock prices from S2 to S10
    for l in list(range(1,lag+1)):
        if isFloat(all_data[cur_index-l][0]):
            dummy.append(float(all_data[cur_index-l][0]))
        else:
            dummy.append(float(0));   # We set this quantity to 0 for now, this will be lagged stock price for S1
        [dummy.append(float(k)) for k in all_data[cur_index-l][1:]]; # Add stock prices at lag l for S2 to S10.
    #print(cur_index, " ", len(dummy))
    return(dummy);        

def feature_names(lag):
    features = [];
    for i in list(range(2,11)):
        features.append('S'+str(i));
    if lag > 0:
        for l in list(range(1,lag+1)):    
            [features.append('S'+str(k)+'_Lag'+str(l)) for k in list(range(1,11))];     
    return features;

def partition_split(alldata,modeldata,testpercent,lag):
  # Step 1: Calculate array dimensions
    n_modeldata = modeldata - lag;   # available model data depends on lag 
    n_features = 9*(lag+1) + lag;  # No. of features =  current + lagged prices of stocks S2 to S10 + lagged prices of stock S1 
  
  # Step 2: Create model and forecast data matrices
    X = np.zeros((n_modeldata,n_features));
    y = np.zeros((n_modeldata,1));
    Xfcast = np.zeros((len(alldata)-modeldata,n_features));
    # Load data
    for j,i in reversed(list(enumerate(alldata))):
        if j > modeldata-1:
            Xfcast[j-modeldata] = extract_features(j,i,alldata,lag);
        
        elif j <= modeldata and j >= lag:
            X[j-lag] = extract_features(j,i,alldata,lag);
            y[j-lag] = i[0];
        #print (j, ' ', y[j]);
    #step 3: Randomly select Training and Testing data
    Xtrain,Xtest,ytrain,ytest = train_test_split(X, y, test_size=testpercent/100, random_state=48)
    #step 4: Feature names
    features = np.array(feature_names(lag));
    return (Xtrain,Xtest,ytrain,ytest,Xfcast,features);


def partition(alldata,modeldata,lag):
  # Step 1: Calculate array dimensions
    n_modeldata = modeldata - lag;   # available model data depends on lag 
    n_features = 9*(lag+1) + lag;  # No. of features =  current + lagged prices of stocks S2 to S10 + lagged prices of stock S1 
  
  # Step 2: Create model and forecast data matrices
    X = np.zeros((n_modeldata,n_features));
    y = np.zeros((n_modeldata,1));
    Xfcast = np.zeros((len(alldata)-modeldata,n_features));
    # Load data
    for j,i in reversed(list(enumerate(alldata))):
        if j > modeldata-1:
            Xfcast[j-modeldata] = extract_features(j,i,alldata,lag);
        
        elif j <= modeldata and j >= lag:
            X[j-lag] = extract_features(j,i,alldata,lag);
            y[j-lag] = i[0];
   # Step 3: Feature Names
    features = feature_names(lag);
    return (X,y,Xfcast,features);

#from fetchdata import fetch as fetch
#alldata = fetch('../data/stock_returns_base150.csv')       
#alldata.pop(0);  
#Xtrain,Xtest,ytrain,ytest,xf = partition(alldata,50,20,4);
#print(Xtrain.shape)
#print(Xtest.shape)  