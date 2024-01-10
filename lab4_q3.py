import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Task 1 Question 3
file_path = 'dataset.csv' 
df = pd.read_csv(file_path)

# Drop the 'MD5' 
df = df.drop(columns=['MD5'])

#features (X) and the target variable (y)
X = df.drop(columns=['label'])  
y = df['label']

#Label Encoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

#SMOTE oversampler
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

#DataFrame from the transformed data
df_resampled = pd.concat([pd.DataFrame(data=X_resampled, columns=X.columns), pd.Series(label_encoder.inverse_transform(y_resampled), name='label')], axis=1)

#counts on the transformed set
class_counts_resampled = df_resampled['label'].value_counts()
print("TransformedDatasetCount:")
print(class_counts_resampled)
