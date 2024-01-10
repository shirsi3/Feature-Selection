import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

# Task 1 question 4
file_path = 'dataset.csv'  
df = pd.read_csv(file_path)

#features (X) and the target variable (y)
X = df.drop(columns=['label'])  
y = df['label']

#RandomUnderSampler
random_undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = random_undersampler.fit_resample(X, y)

#DataFrame from the transformed data
df_resampled = pd.concat([pd.DataFrame(data=X_resampled, columns=X.columns), pd.Series(y_resampled, name='label')], axis=1)

#counts on the transformed set
class_counts_resampled = df_resampled['label'].value_counts()
print("TransformedDatasetCount:")
print(class_counts_resampled)
