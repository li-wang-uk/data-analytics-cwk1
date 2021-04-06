import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


mushroom_data = pd.read_csv("mushrooms.csv")
trial = pd.get_dummies(data=mushroom_data, columns = mushroom_data.columns)

#then we find out which has stalk-root_? is 1, we delete all the stalk-root values as missing, as we dont know what stalk-root is it 
trial.loc[trial['stalk-root_?'] == 1, 'stalk-root_r' ] = np.NaN
trial.loc[trial['stalk-root_?'] == 1, 'stalk-root_b' ] = np.NaN
trial.loc[trial['stalk-root_?'] == 1, 'stalk-root_c' ] = np.NaN
trial.loc[trial['stalk-root_?'] == 1, 'stalk-root_z' ] = np.NaN
trial.loc[trial['stalk-root_?'] == 1, 'stalk-root_u' ] = np.NaN
trial.loc[trial['stalk-root_?'] == 1, 'stalk-root_e' ] = np.NaN
trial['stalk-root_?'] = trial['stalk-root_?'].replace([1], np.nan)
#as the giving data, there is no one has z/u, but it is a type, we have to assgin stalk-root_u and stalk-root_z to 0, where there is a type for stalk-root 
trial.loc[trial['stalk-root_r'] == 1, 'stalk-root_u' ] = 0
trial.loc[trial['stalk-root_b'] == 1, 'stalk-root_u' ] = 0
trial.loc[trial['stalk-root_c'] == 1, 'stalk-root_u' ] = 0
trial.loc[trial['stalk-root_e'] == 1, 'stalk-root_u' ] = 0
trial.loc[trial['stalk-root_r'] == 1, 'stalk-root_z' ] = 0
trial.loc[trial['stalk-root_b'] == 1, 'stalk-root_z' ] = 0
trial.loc[trial['stalk-root_c'] == 1, 'stalk-root_z' ] = 0
trial.loc[trial['stalk-root_e'] == 1, 'stalk-root_z' ] = 0
#then we delete the ? column, as we are going to use KNN method to give a value 
trial1= trial.drop(columns = ['stalk-root_?'])

results1 = []
results2 = []
results3 = []
results4 = []

def KNN_test(n):
    #Applying KNN
    imputer = KNNImputer(n_neighbors=n)
    trial_imputed = imputer.fit_transform(trial1)
    trial_imputed = pd.DataFrame(trial_imputed, columns = trial1.columns)
    for col in trial_imputed.columns:
        trial_imputed[col] = trial_imputed[col].round()

    x = trial_imputed.iloc[:,2:].values
    y = trial_imputed.iloc[:,1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)
    #Applying PCA
    pca = PCA(n_components = 5)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    
    # Training the Logistic Regression Model on the Training set
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(x_train, y_train)
    # Predicting the test set
    y_pred = classifier.predict(x_test)
    # Calculating accuracy score
    results1.append(accuracy_score(y_test, y_pred))
    # Calculating cross validation score
    score = cross_val_score(classifier,x,y,cv=5)
    results1.append(f'Cross validation Accuaracy:  {np.mean(score)}')

    #Training the Random Forest Classification on the Training set
    classifier = RandomForestClassifier(criterion = 'entropy', random_state = 0, n_estimators = 100)
    classifier.fit(x_train, y_train)
    # Predicting the test set 
    y_pred = classifier.predict(x_test)
    results2.append(accuracy_score(y_test, y_pred))
    # Calculating cross validation score
    score = cross_val_score(classifier,x,y,cv=5)
    results2.append(f'Cross validation Accuaracy:  {np.mean(score)}')

    #Training the SVC Classifier on the Training set
    classifier = Pipeline([('scaler', StandardScaler()),('svc', SVC())])
    classifier.fit(x_train, y_train)
    # Predicting the test set 
    y_pred = classifier.predict(x_test)
    # Calculating accuracy score
    results3.append(accuracy_score(y_test, y_pred))
    # Calculating cross validation score
    score = cross_val_score(classifier,x,y,cv=5)
    results3.append(f'Cross validation Accuaracy:  {np.mean(score)}')

    #Training the Decision Tree Classifier on the Training set
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(x_train, y_train)
    # Predicting the test set 
    y_pred = classifier.predict(x_test)
    results4.append(accuracy_score(y_test, y_pred))
    # Calculating cross validation score
    score = cross_val_score(classifier,x,y,cv=5)
    results4.append(f'Cross validation Accuaracy:  {np.mean(score)}')

def KNN_optimal_test(n):
    for i in range(1, n +1, 2):
        KNN_test(i)
    print(max(results1),results1.index(max(results1))+1)
    print(max(results2),results2.index(max(results2))+1)

