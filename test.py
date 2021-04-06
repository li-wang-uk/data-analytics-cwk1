import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from chefboost import Chefboost as chef

mushroom_file_path ="mushrooms.csv"
mushroom_data = pd.read_csv(mushroom_file_path)

# Creating independent and dependent variables
x = mushroom_data.iloc[:,1:]
y = mushroom_data.iloc[:,0]
# Label encoding
le = LabelEncoder()
y = le.fit_transform(y)
onehotencoder = OneHotEncoder()
x = onehotencoder.fit_transform(x).toarray()
# Splitting the dataset into training set and test set

results1 = []
results2 = []
results3 = []
results4 = []

def pca_ncompnents_test(n):
    # Applying PCA
    pca = PCA(n_components = n)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)
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
    # Calculating accuracy score
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

def PCA_optimal_test(n):
    for i in range(1,n+1):
        pca_ncompnents_test(i)
    print(max(results1),results1.index(max(results1))+1)
    print(max(results2),results2.index(max(results2))+1)

# Creating C4.5 decision tree model

config = {'algorithm': 'C4.5'}

decisiondata = mushroom_data.copy()

last_col = decisiondata.pop('class')
decisiondata.insert(22, 'Decision', last_col)
decisiondata.to_csv('decisiondata.csv', index = False)

#Setting 80% train data and 20% test data (keeping x and y together)
dectest = decisiondata[-1625:]
dectrain = decisiondata[0:6499]

#Saving the model 
C45model = chef.fit(dectrain, config = config)
chef.save_model(C45model, 'model.pkl')

prediction = []
actual = []
for index, instance in dectest.iterrows():
    prediction += chef.predict(C45model, instance)
    actual += instance['Decision']
print(accuracy_score(prediction, actual))
#C45model = chef.load_model('model.pkl') to load model
