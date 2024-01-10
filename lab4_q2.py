import pandas as pd
from imblearn.over_sampling import RandomOverSampler

# Task 1 Question 2
file_path = 'dataset.csv' 
df = pd.read_csv(file_path)

#features (X) and the target variable (y)
X = df.drop(columns=['label'])  
y = df['label']

#RandomOverSampler
random = RandomOverSampler(random_state=42)
X_resampled, y_resampled = random.fit_resample(X, y)

# making data frame
df_resampled = pd.DataFrame(data=X_resampled, columns=X.columns)
df_resampled['label'] = y_resampled

# Transformed data set
class_counts_resampled = df_resampled['label'].value_counts()
print("TransformedDatasetCount:")
print(class_counts_resampled)
