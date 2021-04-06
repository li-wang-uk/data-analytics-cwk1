#Import 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer

#Data Cleaning 
#2 Different Methods to deal with missing data 

#Method 1- Keep the missing data "?" as a question marks 
#doing nothing on the dataset 

#Method 2- Use KNN method to give values for missing data "?"
#split each value of the columns into new columns. E,g, there was only one column called stalk-root, here we split it into new columns as stalk-root_?, stalk-root_b, stalk-root_c, stalk-root_u,stalk-root_e,stalk-root_z, stalk-root_r
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
imputer = KNNImputer(n_neighbors=5) # maybe we change the neighbours
trial_imputed = imputer.fit_transform(trial1)
trial_imputed = pd.DataFrame(trial_imputed, columns = trial1.columns)
for col in trial_imputed.columns:
    trial_imputed[col] = trial_imputed[col].round()
trial_imputed.to_csv('b.csv') 


