# Main project file for stock price prediction
from fetchdata import fetch
from partitiondata import partition_split
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from sklearn.linear_model.ridge import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import RFE 
from sklearn.feature_selection import RFECV
# Load CSV file    
alldata = fetch('../data/stock_returns_base150.csv')       
alldata.pop(0);             # Remove the header 
print(len(alldata));

# Partition data into separate objects for training, testing and forecasting. Also generate the features matrix X and output y 
lag = 3;
print("Lag is ", lag); 
testpercent = 25;
n_modeldata = 50;

X,Xtest,y,ytest,Xfcast,features = partition_split(alldata,n_modeldata,testpercent,lag);
print(' X is ', X.shape , '\t Xtest is ', Xtest.shape, '\t Xfcast is ', Xfcast.shape,
      '\n y is ',y.shape , '\t \t ytest is ', ytest.shape, '\n features is ', features.shape);
n_features = features.shape[0];

# Features Ranking
# Univariate Feature Scoring with Cross validation 


def plotscores(scores,rf,n = n_features):
    fig, ax1 = plt.subplots(figsize=(10, 18))
    plt.subplots_adjust(left=0.1, right=0.9)
    fig.canvas.set_window_title('Feature Importances - ' + rf._estimator_type)
    ybar = [k[0] for k in scores];
    xbar = [k[1] for k in scores];
    pos = np.arange(len(xbar)) + 0.5 ; # num_items = len(xbar)
    ax1.axis([-1, 1, -1, n_features+1]);
    ax1.barh(pos, ybar, align='center', height=0.5, color='m')
    plt.yticks(pos, xbar);
    plt.title('Cross validated feature scores (R^2)');
    plt.savefig('featurerank.png')
    plt.show()
    
# Select Regularization parameter alpha using RidgeCV, which does cross validation on model data to determine the best alpha.
y = y.reshape(len(y),);
rcv = RidgeCV(alphas=[10,20,30,40,50]); 
rcv.fit(X,y);
#print('rcv score = ', rcv.score(X,y));
print('alpha selected from cv is ', rcv.alpha_);

# Construct an estimator using the best alpha and rank features 
from sklearn.cross_validation import cross_val_score, ShuffleSplit
def rankfeatures(X,Y,rf,names):
    scores = []
    for i in range(X.shape[1]):
        score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2" ,cv=ShuffleSplit(len(X), 20, .2));
        scores.append((np.mean(score),names[i]));
   # for i in (sorted(scores, reverse=True)):
   #     print(i[1], ' ', round(i[0],2));
    return sorted(scores); 

rf = Ridge(alpha=rcv.alpha_);     
scores = rankfeatures(X,y,rf,features);
plotscores(scores,rf);


# Model Selection
def plotfit(model,Xtest,ytest,c = 'red', title = 'Fit model'):
    y_sm = np.array(model.predict(Xtest));    
    x_sm = np.array(list(range(1,len(ytest)+1)));
    x_smooth = np.linspace(x_sm.min(), x_sm.max(), 200)
    y_smooth = spline(x_sm, y_sm, x_smooth);    
    plt.plot(x_smooth, y_smooth, color=c, linewidth=3)
    plt.scatter(x_sm, ytest, color='black')
    plt.xlabel('Samples')
    plt.ylabel('Returns')
    plt.title(title)
    plt.show()
    
# Regressive Feature Elimination, This will help to eliminate over-fitting by eliminating irrelevant and highly correlated features     
selector1 = RFECV(rf,cv=5);
selector1 = selector1.fit(X,y);
print("Features selected by RFECV ", selector1.n_features_);

# RFECV yields a model with low number (4~5) of features with K-fold cross validation with 5 folds.
# Since our sample size is small, additional features are selected to avoid  over-fitting on the model data.
# 8 Features were selected using RFE.

selector2 = RFE(rf,8); # Recursive feature elimination to select 8 best features. 
selector2 = selector2.fit(X,y);
print (selector2.n_features_)
for i,j in enumerate(selector2.support_):
    if j == True:
        print(features[i])
predictor = selector2.estimator_;
print('Variance score Train: %.2f' % selector2.score(X,y));
print('Variance score Test: %.2f' % selector2.score(Xtest,ytest));
print('Coeff of Test: ', selector2.ranking_);
print('No of Features selected by RFE = %.2f' %sum(selector2.support_));

plotfit(selector2,X,y, title = 'Training fit');
plotfit(selector2,Xtest,ytest,c='blue', title = 'Test fit');


# Forecast, 
# Since lagged variables are selected for our model, forecasting is done iteratively 
# by using the predicted values at time t as lagged variables for time t+1,t+2...
# In practice however, this is not required as the true price will be known before prediction.
yfcast = [];
for i in  list(range(len(Xfcast))):
    Xfcast_trim = selector2.transform(Xfcast);
    x = Xfcast_trim[i,:];
    x = x.reshape(1,-1);
    ypred = predictor.predict(x);
    yfcast.append(ypred);
    k = 9;j=1;
    if i < len(Xfcast)-1:
        for l in list(range(0,lag)): 
            #print ('l = ', l , 'i+j = ', i+j, ' k+1 = ', k+1);
            if i+j <= 49:
                Xfcast[i+j,k] = ypred;
            k = k+10;
            j = j+1;
#np.savetxt('../Xfcast-loaded.csv', Xfcast, fmt="%f", delimiter = ',');            

# Write predictions
from fetchdata import futuredates, write_fcast
dates = futuredates('../data/stock_returns_base150.csv');
import itertools
chain1 = itertools.chain(*dates)
chain2 = itertools.chain(*yfcast)
dates  = list(chain1);
yfcast = list(chain2)
out = []
for i,j in zip(dates,yfcast):
    out.append([i,j]);
write_fcast('../prediction.csv',out);