import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

file_path = 'dataset.csv'
df = pd.read_csv(file_path)

# Drop the 'MD5' column
df = df.drop(columns=['MD5'])

#features (X) and target variable (y)
X = df.drop(columns=['label'])  
y = df['label']

# select 15 feartures
k = 15

#chi2 selector
chi2_selector = SelectKBest(chi2, k=k)
X_chi2 = chi2_selector.fit_transform(X, y)

#names of selected features using chi2
selected_features_chi2 = X.columns[chi2_selector.get_support()]

#mutual information
mi_selector = SelectKBest(mutual_info_classif, k=k)
X_mi = mi_selector.fit_transform(X, y)

#names of selected features 
selected_features_mi = X.columns[mi_selector.get_support()]


print("selected using chi2:")
print(selected_features_chi2)
print("\nselected using mutual information:")
print(selected_features_mi)
