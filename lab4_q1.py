import pandas as pd

# Task1 Question 1
df = pd.read_csv('dataset.csv')

# Replace column name with label
class_column_name = 'label'

# count
class_counts = df[class_column_name].value_counts()
print("Class Counts:")
print(class_counts)
