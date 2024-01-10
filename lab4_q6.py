import pandas as pd
from sklearn.feature_selection import SelectPercentile, chi2, mutual_info_classif

#dataset 
file_path = 'dataset.csv'
df = pd.read_csv(file_path)

# Drop the 'MD5'
df = df.drop(columns=['MD5'])

#features (X) and target variable (y)
X = df.drop(columns=['label'])  
y = df['label']

# Percentage of features
percentile = 10

#chi2
chi2_selector = SelectPercentile(chi2, percentile=percentile)
X_chi2 = chi2_selector.fit_transform(X, y)

# names of selected features for chi2
selected_features_chi2 = X.columns[chi2_selector.get_support()]

#mutual information
mi_selector = SelectPercentile(mutual_info_classif, percentile=percentile)
X_mi = mi_selector.fit_transform(X, y)

#names of selected features for mutual information
selected_features_mi = X.columns[mi_selector.get_support()]

print("selected using chi2:")
print(selected_features_chi2)
print("\nselected using mutal information")
print(selected_features_mi)
