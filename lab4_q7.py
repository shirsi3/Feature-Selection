import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

#dataset
file_path = 'dataset.csv'
df = pd.read_csv(file_path)

# Drop the 'MD5' 
df = df.drop(columns=['MD5'])

#features (X) and target variable (y)
X = df.drop(columns=['label'])  
y = df['label']

# features selected is 15
n_features_to_select = 15

#SVC classifier
svc = SVC(kernel="linear")

#RFE
rfe = RFE(estimator=svc, n_features_to_select=n_features_to_select, step=1)
X_rfe = rfe.fit_transform(X, y)

#names of selected features
selected_features_rfe = X.columns[rfe.support_]

print("features using RFE:")
print(selected_features_rfe)
