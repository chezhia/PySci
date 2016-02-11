from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn import datasets
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

def featureselect(X,y):
   # Recursive Feature Elimination
# load the iris datasets
    #dataset = datasets.load_iris()
# create a base classifier used to evaluate a subset of attributes
 
    model = Ridge(alpha = 1.0);
# create the RFE model and select 3 attributes
    rfe = RFECV(model,10);
    rfe = rfe.fit(X, y)
    print(rfe.alpha_)
    # summarize the selection of the attributes
    print(rfe.support_)
    print(rfe.ranking_)
    
# Average Ranking using all methods
def rank_to_dict(ranks, names, order=1):
        minmax = MinMaxScaler()
        ranks = np.array(ranks);
        ranks = ranks.astype(float);
        ranks = ranks.reshape(-1,1);
        ranks = minmax.fit_transform(ranks);
        ranks = [float(str(round(i,3))) for i in ranks[:,0]];
        d = dict(zip(names, ranks))
        return d



def RankFeatures(X,Y,names):
    Y = Y.reshape(len(Y),)
    ranks = {};
    lr = LinearRegression(normalize=True)
    lr.fit(X, Y)
    ranks["Linear.reg"] = rank_to_dict(np.abs(lr.coef_), names);
     
    ridge = Ridge(alpha=7)
    ridge.fit(X, Y)
    ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)
     
    lasso = Lasso(alpha=.05)
    lasso.fit(X, Y)
    ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), names)
     
     
    rlasso = RandomizedLasso(alpha=0.04)
    rlasso.fit(X, Y)
    ranks["Stability"] = rank_to_dict(np.abs(rlasso.scores_), names)
     
    #stop the search when 5 features are left (they will get equal scores)
    rfe = RFE(lr, n_features_to_select=5)
    rfe.fit(X,Y)
    ranks["RFE"] = rank_to_dict([float(i) for i in rfe.ranking_], names, order=-1)
     
    rf = RandomForestRegressor()
    rf.fit(X,Y)
    ranks["RF"] = rank_to_dict(rf.feature_importances_, names)
     
     
    f, pval  = f_regression(X, Y, center=True)
    ranks["Corr."] = rank_to_dict(f, names) 
    r = {}
    for name in names:
        r[name] = float(str(round(np.mean([ranks[method][name] for method in ranks.keys()]), 3)));
    print(r);
    methods = sorted(ranks.keys());
    ranks["Mean"] = r;
    methods.append("Mean");
    print ("\t%s" % "\t".join(methods));
    
    for name in names:
        print ("%s\t%s" % (name, "\t".join(map(str, 
                             [ranks[method][name] for method in methods]))))
    
RankFeatures(X,y,features);


def simplereg(Xtrain,Xtest,ytrain,ytest):
    clf = LinearRegression(); # LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    clf.fit(Xtrain,ytrain);
    #print('Coefficients: \n', clf.coef_)
            # The mean square error
    print("Residual sum of squares: %.2f"
         % np.mean((clf.predict(Xtest) - ytest) ** 2));
        # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % clf.score(Xtest, ytest));
    print('\n');
 

# Regularized Linear Regression

def regularizedreg(Xtrain,Xtest,ytrain,ytest):
    Rclf = RidgeCV(alphas=[1,2,20,40,50]) # RidgeCV(alphas=[0.1, 1.0, 2.0, 4.0, 20.0], cv=None, fit_intercept=True, scoring=None, normalize=False)
    Rclf.fit(Xtrain,ytrain);
    print("Residual sum of squares: %.2f"
         % np.mean((Rclf.predict(Xtest) - ytest) ** 2))
    print('Regularization choosen, alpha = %.2f' % Rclf.alpha_);
    print(' Coef values = ', Rclf.coef_);                                      
    print('Variance score: %.2f' % Rclf.score(Xtest, ytest))


selector = RFECV(rf, step = 1, cv=ShuffleSplit(len(X), 10, .2))
selector = selector.fit(X,y);
print (selector.n_features_)
for i,j in enumerate(selector.support_):
    if j == True:
        print(features[i])

print('Variance score Train: %.2f' % selector.score(X,y));
print('Variance score Test: %.2f' % selector.score(Xtest,ytest));
#print('Coeff of Test: ', selector.coef_);
print('No of Features selected by RFECV = %.2f' %sum(selector.support_));
plotfit(selector,Xtest,ytest);

# Learning Curve
from sklearn.learning_curve import learning_curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes= list(range(3,23,3))):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



title = "Learning Curves (Ridge)"

plot_learning_curve(rf, title, X, y, ylim=(0.7, 1.01))

plt.show()
    
# Validation Curve
if 0:
    from sklearn.learning_curve import validation_curve
    param_range = np.array(list(range(0,50,10)));
    train_scores, test_scores = validation_curve(Ridge(), X, y, "alpha", param_range, cv=ShuffleSplit(len(X), 20 , 0.3))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.axis([0,50,0.0, 1.1]);
    plt.plot(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    plt.title("Validation Curve with Ridge")
    plt.xlabel("alpha")
    plt.ylabel("Score")
    plt.show()
    
    
#simplereg(X,Xtest,y,ytest);
#regularizedreg(X,Xtest,y,ytest);
#from selectfeatures import featureselect
#y = y.reshape(len(y),)
#featureselect(X