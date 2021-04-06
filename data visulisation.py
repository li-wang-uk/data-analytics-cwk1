#Data Visulisaiton 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
mushroom_data = pd.read_csv("mushrooms.csv")
import math
from collections import Counter
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import sklearn.preprocessing as sp
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from subprocess import check_output
from sklearn import tree


#Comparison poisonus and edible mushrooms (BAR CHART)
plt.figure(figsize=(10,5))
plt.title('Poisonous and edible mushrooms', fontsize=14)
sns.countplot(x='class', data=mushroom_data, palette=('#9b111e', '#50c878'))
plt.xlabel("Mushroom Type", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

#quantity of values for each columns (BAR CHART)
fig = plt.figure(figsize=(20,15))
ax = fig.gca()
mush_encoded.hist(ax=ax) #need to be changed to the name of encoded csv file
plt.show()

#Heat Map Correlation 
plt.figure(figsize=(12,10))
ax = sns.heatmap(mush_encoded.corr()) #need to be changed to the name of encoded csv file

#histogram for every column based on class 
#First 6 plots combined
plt.figure(figsize = (12,10))
ax1 = plt.subplot2grid(shape = (3,3), loc = (0,0))
capshapept= sns.histplot(x='class',hue='cap-shape', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax1 = capshapept
ax1.set_xlabel('')
ax1.set_title('cap-shape')

ax2 = plt.subplot2grid(shape = (3,3), loc = (0,1))
capsurfacept= sns.histplot(x='class',hue='cap-surface', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax2 = capsurfacept
ax2.set_xlabel('')
ax2.set_title('cap-surface')

ax3 = plt.subplot2grid(shape = (3,3), loc = (0,2))
capcolorpt= sns.histplot(x='class',hue='cap-color', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax3 = capcolorpt
ax3.set_xlabel('')
ax3.set_title('cap-color')

ax4 = plt.subplot2grid(shape = (3,3), loc = (1,0))
bruisespt= sns.histplot(x='class',hue='bruises', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax4 = bruisespt
ax4.set_xlabel('')
ax4.set_title('bruises')

ax5 = plt.subplot2grid(shape = (3,3), loc = (1,1))
odorpt= sns.histplot(x='class',hue='odor', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax5 = odorpt
ax5.set_xlabel('')
ax5.set_title('odor')


ax6 = plt.subplot2grid(shape = (3,3), loc = (1,2))
gillattachementpt= sns.histplot(x='class',hue='gill-attachment', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax6 = gillattachementpt
ax6.set_xlabel('')
ax6.set_title('gill-attachment')
plt.show()

# second combined plot 
plt.figure(figsize = (12,10))
ax7 = plt.subplot2grid(shape = (3,3), loc = (0,0))
gillspacingpt= sns.histplot(x='class',hue='gill-spacing', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax7 = gillspacingpt
ax7.set_xlabel('')
ax7.set_title('gill-spacing')

ax8 = plt.subplot2grid(shape = (3,3), loc = (0,1))
gillsizept= sns.histplot(x='class',hue='gill-size', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax8 = gillsizept
ax8.set_xlabel('')
ax8.set_title('gill-size')

ax9 = plt.subplot2grid(shape = (3,3), loc = (0,2))
gillcolorpt= sns.histplot(x='class',hue='gill-color', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax9 = gillcolorpt
ax9.set_xlabel('')
ax9.set_title('gill-color')

ax10 = plt.subplot2grid(shape = (3,3), loc = (1,0))
stalkshapept= sns.histplot(x='class',hue='stalk-shape', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax10 = stalkshapept
ax10.set_xlabel('')
ax10.set_title('stalk-shape')

ax11 = plt.subplot2grid(shape = (3,3), loc = (1,1))
stalksurfaceaboveringpt= sns.histplot(x='class',hue='stalk-surface-above-ring', data=mushroom_data,multiple = 'stack', shrink =0.1)
ax11 = stalksurfaceaboveringpt
ax11.set_xlabel('')
ax11.set_title('stalk-surface-above-ring')


ax12 = plt.subplot2grid(shape = (3,3), loc = (1,2))
stalksurfacebelowringpt= sns.histplot(x='class',hue='stalk-surface-below-ring', data=mushroom_data,multiple = 'stack', shrink =0.1)
ax12 = stalksurfacebelowringpt
ax12.set_xlabel('')
ax12.set_title('stalk-surface-below-ring')
plt.show()

#third combined plot 
plt.figure(figsize = (12,10))
ax13 = plt.subplot2grid(shape = (3,3), loc = (0,0))
stalkcoloraboveringpt= sns.histplot(x='class',hue='stalk-color-above-ring', data=mushroom_data,multiple = 'stack', shrink =0.1)
ax13 = stalkcoloraboveringpt
ax13.set_xlabel('')
ax13.set_title('stalk-color-above-ring')

ax14 = plt.subplot2grid(shape = (3,3), loc = (0,1))
stalkcolorbelowringpt= sns.histplot(x='class',hue='stalk-color-below-ring', data=mushroom_data,multiple = 'stack', shrink =0.1)
ax14 = stalkcolorbelowringpt
ax14.set_xlabel('')
ax14.set_title('stalk-color-below-ring')

ax15 = plt.subplot2grid(shape = (3,3), loc = (0,2))
veiltypept= sns.histplot(x='class',hue='veil-type', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax15 = veiltypept
ax15.set_xlabel('')
ax15.set_title('veil-type')

ax16 = plt.subplot2grid(shape = (3,3), loc = (1,0))
veilcolorpt= sns.histplot(x='class',hue='veil-color', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax16 = veilcolorpt
ax16.set_xlabel('')
ax16.set_title('veil-color')

ax17 = plt.subplot2grid(shape = (3,3), loc = (1,1))
ringnumberpt= sns.histplot(x='class',hue='ring-number', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax17 = ringnumberpt
ax17.set_xlabel('')
ax17.set_title('ring-number')


ax18 = plt.subplot2grid(shape = (3,3), loc = (1,2))
ringtypept= sns.histplot(x='class',hue='ring-type', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax18 = ringtypept
ax18.set_xlabel('')
ax18.set_title('ring-type')
plt.show()

#fourth combined 
plt.figure(figsize = (12,10))
ax19 = plt.subplot2grid(shape = (3,3), loc = (0,0))
sporeprintcolorpt= sns.histplot(x='class',hue='spore-print-color', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax19 = sporeprintcolorpt
ax19.set_xlabel('')
ax19.set_title('spore-print-color')

ax20 = plt.subplot2grid(shape = (3,3), loc = (0,1))
populationpt= sns.histplot(x='class',hue='population', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax20 = populationpt
ax20.set_xlabel('')
ax20.set_title('population')

ax21 = plt.subplot2grid(shape = (3,3), loc = (0,2))
habitatpt= sns.histplot(x='class',hue='habitat', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax21 = habitatpt
ax21.set_xlabel('')
ax21.set_title('habitat')
plt.show()
#the column with missing value 
stalkrootpt= sns.histplot(x='class',hue='stalk-root', data=mushroom_data,multiple = 'stack', shrink =0.3) #missing data as a factor 
#histogram for 'class' based on every column 
#First combined plot 
plt.figure(figsize = (12,10))
ax22 = plt.subplot2grid(shape = (3,3), loc = (0,0))
capshapept1= sns.histplot(x='cap-shape',hue='class', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax22 = capshapept1
ax22.set_xlabel('')
ax22.set_title('cap-shape')

ax23 = plt.subplot2grid(shape = (3,3), loc = (0,1))
capsurfacept1= sns.histplot(x='cap-surface',hue='class', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax23 = capsurfacept1
ax23.set_xlabel('')
ax23.set_title('cap-surface')

ax24 = plt.subplot2grid(shape = (3,3), loc = (0,2))
capcolorpt1= sns.histplot(x='cap-color',hue='class', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax3 = capcolorpt1
ax3.set_xlabel('')
ax3.set_title('cap-color')

ax25 = plt.subplot2grid(shape = (3,3), loc = (1,0))
bruisespt1= sns.histplot(x='bruises',hue='class', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax25 = bruisespt1
ax25.set_xlabel('')
ax25.set_title('bruises')

ax26 = plt.subplot2grid(shape = (3,3), loc = (1,1))
odorpt1= sns.histplot(x='odor',hue='class', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax26 = odorpt1
ax26.set_xlabel('')
ax26.set_title('odor')

ax27 = plt.subplot2grid(shape = (3,3), loc = (1,2))
gillattachementpt1= sns.histplot(x='gill-attachment',hue='class', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax27 = gillattachementpt1
ax27.set_xlabel('')
ax27.set_title('gill-attachment')
plt.show()
#Second combined plot 
plt.figure(figsize = (12,10))
ax28 = plt.subplot2grid(shape = (3,3), loc = (0,0))
gillspacingpt1= sns.histplot(x='gill-spacing',hue='class', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax28 = gillspacingpt1
ax28.set_xlabel('')
ax28.set_title('gill-spacing')

ax29 = plt.subplot2grid(shape = (3,3), loc = (0,1))
gillsizept1= sns.histplot(x='gill-size',hue='class', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax29 = gillsizept1
ax29.set_xlabel('')
ax29.set_title('gill-size')

ax30 = plt.subplot2grid(shape = (3,3), loc = (0,2))
gillcolorpt1= sns.histplot(x='gill-color',hue='class', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax30 = gillcolorpt1
ax30.set_xlabel('')
ax30.set_title('gill-color')

ax31 = plt.subplot2grid(shape = (3,3), loc = (1,0))
stalkshapept1= sns.histplot(x='stalk-shape',hue='class', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax31 = stalkshapept1
ax31.set_xlabel('')
ax31.set_title('stalk-shape')

ax32 = plt.subplot2grid(shape = (3,3), loc = (1,1))
stalksurfaceaboveringpt1= sns.histplot(x='stalk-surface-above-ring',hue='class', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax32 = stalksurfaceaboveringpt1
ax32.set_xlabel('')
ax32.set_title('stalk-surface-above-ring')

ax33 = plt.subplot2grid(shape = (3,3), loc = (1,2))
stalksurfacebelowringpt1= sns.histplot(x='stalk-surface-below-ring',hue='class', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax33 = stalksurfacebelowringpt1
ax33.set_xlabel('')
ax33.set_title('stalk-surface-below-ring')
plt.show()
#Third combined plot 
plt.figure(figsize = (12,10))
ax34 = plt.subplot2grid(shape = (3,3), loc = (0,0))
stalkcoloraboveringpt1= sns.histplot(x='stalk-color-above-ring',hue='class', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax34 = stalkcoloraboveringpt1
ax34.set_xlabel('')
ax34.set_title('stalk-color-above-ring')

ax35 = plt.subplot2grid(shape = (3,3), loc = (0,1))
stalkcolorbelowringpt1= sns.histplot(x='stalk-color-below-ring',hue='class', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax35 = stalkcolorbelowringpt1
ax35.set_xlabel('')
ax35.set_title('stalk-color-below-ring')

ax36 = plt.subplot2grid(shape = (3,3), loc = (0,2))
veiltypept1= sns.histplot(x='veil-type',hue='class', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax36 = veiltypept1
ax36.set_xlabel('')
ax36.set_title('veil-type')

ax37 = plt.subplot2grid(shape = (3,3), loc = (1,0))
veilcolorpt1= sns.histplot(x='veil-color',hue='class', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax37 = veilcolorpt1
ax37.set_xlabel('')
ax37.set_title('veil-color')

ax38 = plt.subplot2grid(shape = (3,3), loc = (1,1))
ringnumberpt1= sns.histplot(x='ring-number',hue='class', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax38 = ringnumberpt1
ax38.set_xlabel('')
ax38.set_title('ring-number')

ax39 = plt.subplot2grid(shape = (3,3), loc = (1,2))
ringtypept1= sns.histplot(x='ring-type',hue='class', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax39 = ringtypept1
ax39.set_xlabel('')
ax39.set_title('ring-type')
plt.show()

#Fourth combined plot 
plt.figure(figsize = (12,10))
ax40 = plt.subplot2grid(shape = (3,3), loc = (0,0))
sporeprintcolorpt1= sns.histplot(x='spore-print-color',hue='class', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax40 = sporeprintcolorpt1
ax40.set_xlabel('')
ax40.set_title('spore-print-color')

ax41 = plt.subplot2grid(shape = (3,3), loc = (0,1))
populationpt1= sns.histplot(x='population',hue='class', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax41 = populationpt1
ax41.set_xlabel('')
ax41.set_title('population')

ax42 = plt.subplot2grid(shape = (3,3), loc = (0,2))
habitatpt1= sns.histplot(x='habitat',hue='class', data=mushroom_data,multiple = 'stack', shrink =0.3)
ax42 = habitatpt1
ax42.set_xlabel('')
ax42.set_title('habitat')

plt.show()
#the column with missing value 
stalkrootpt1= sns.histplot(x='stalk-root',hue='class', data=mushroom_data,multiple = 'stack', shrink =0.3) 

#Transfer data after machine learning (Which only has 0 and 1) back to non-numeric letter data for column ‘Stalk-root’ which has missing value. 
trial_imputed = pd.read_csv('b.csv')
aaa = trial_imputed.loc[:, trial_imputed.columns.str.startswith('class')|trial_imputed.columns.str.startswith('stalk-root')]
aaa.columns=['class_e','class_p','stalkroot_b','stalkroot_c','stalkroot_e','stalkroot_r','stalkroot_z','stalkroot_u']
def change_to_letter_stalkroot(b,u,e,r,z,c):
    if b == 1:
        return 'b'
    elif u ==1:
        return 'u'
    elif e == 1:
        return 'e'
    elif r == 1:
        return 'r'
    elif z ==1:
        return 'z'
    else:
        return 'c'
def change_to_letter_class(class_e):
    if class_e == 1:
        return 'e'
    else:
        return 'p'
aaa['class']=aaa.apply(lambda x: change_to_letter_class(x.class_e), axis =1 )
aaa['stalk-root']=aaa.apply(lambda x: change_to_letter_stalkroot(x.stalkroot_b,x.stalkroot_u,x.stalkroot_e,x.stalkroot_r,x.stalkroot_z,x.stalkroot_c), axis =1 )
#histogram of 'stalk-root' based on value of the column
stalkrootknnpt= sns.histplot(x='class',hue='stalk-root', data=aaa,multiple = 'stack', shrink =0.3)
#histogram of 'stalk-root' based on 'class'
stalkrootknnpt1= sns.histplot(x='stalk-root',hue='class', data=aaa,multiple = 'stack', shrink =0.3)

#theil's U uncertainty coefifcient heatmap

def conditional_entropy(x,y):
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy

def theil_u(x,y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x

theilu = pd.DataFrame(index=['class'],columns=mushroom_data2.columns)
columns = mushroom_data2.columns
for j in range(0,len(columns)):
    u = theil_u(mushroom_data2['class'].tolist(),mushroom_data2[columns[j]].tolist())
    theilu.loc[:,columns[j]] = u
theilu.fillna(value=np.nan,inplace=True)
plt.figure(figsize=(20,1))
sns.heatmap(theilu,annot=True,fmt='.2f')
plt.show()

# Crammer's V correlation
mushroom_file_path ="mushrooms.csv"
mushroom_data = pd.read_csv(mushroom_file_path)
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

cor_list = []
cor_names = list(mushroom_data.columns)
cor_names.pop(0)

def cor_test():
    y = mushroom_data.iloc[:,0]
    for i in range(1,len(cor_names)+1):
        x = mushroom_data.iloc[:,i]
        cor = round(cramers_v(x,y),2)
        cor_list.append(cor)

cor_test()

fi_df = pd.DataFrame({
    "cramer's V correlations" : cor_list,
    "features" : cor_names
})

fi_df.sort_values(by="cramer's V correlations", ascending=False, inplace=True)

plt.figure(figsize=(10,7))
sns.barplot(x="cramer's V correlations", y="features", palette="twilight", data=fi_df)
plt.show()

#Accuracy model comparison(bar chart)
acscore =  [1.0,0.9993846153846154,0.984,0.932923076923077]
models = ["Random Forest Model(approach 1)","Random Forest Model(approach 2)", "Logistic Regression Model(approach 1)","Logistic Regression Model(approach 2)"]

plt.rcParams['figure.figsize']=15,8 
plt.style.use('dark_background')
ax = sns.barplot(x=models, y=acscore, palette = "rocket", saturation =1.5)
plt.xlabel("Classifier Models", fontsize = 20 )
plt.ylabel("% of Accuracy", fontsize = 20)
plt.title("Accuracy of different Classifier Models", fontsize = 20)
plt.xticks(fontsize = 10, horizontalalignment = 'center', rotation = 0)
plt.yticks(fontsize = 13)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()

#Random Forest' decision tree graph
import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree


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
imputer = KNNImputer(n_neighbors=5) 
trial_imputed = imputer.fit_transform(trial1)
trial_imputed = pd.DataFrame(trial_imputed, columns = trial1.columns)
for col in trial_imputed.columns:
    trial_imputed[col] = trial_imputed[col].round()
x = trial_imputed.iloc[:,2:]
y = trial_imputed.iloc[:,1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)
#Training the Random Forest Classification on the Training set
classifier = RandomForestClassifier(criterion = 'entropy', random_state = 0, n_estimators = 100)
classifier.fit(x_train, y_train)
# Predicting the test set 

y_pred = classifier.predict(x_test)

estimator = classifier.estimators_[5]

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (20,20), dpi=800)
tree.plot_tree(classifier.estimators_[0],
            feature_names = x.columns,
            class_names = ["poison","edible"],
            filled = True)
fig.savefig('rf_individualtree.png')


