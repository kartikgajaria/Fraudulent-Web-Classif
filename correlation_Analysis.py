import pandas as pd
from sklearn.datasets import load_diabetes
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
from scipy.stats import chi2_contingency
# Basic packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
# Sklearn modules & classes
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


"""df = pd.read_csv('finalDataset-FixedMissingValues.csv')

print(df.head())

# Find the pearson correlations matrix
corr = df.corr(method = 'pearson')
print(corr)

plt.figure(figsize=(10,8), dpi =100)
sns.heatmap(corr,annot=True,fmt=".2f", linewidth=.5)
#plt.show()

"""

#data = pd.read_csv('finalDataset-FixedMissingValues-pandasT - Copy.csv')
data = pd.read_csv('new_dataset_corrected.csv')         #uncoment this  <------- PRIM


cols_to_scale = ['external_links', 'internal_links','age','total_life']
scaler = StandardScaler()
scaler.fit_transform(data[cols_to_scale])
data[cols_to_scale] = scaler.transform(data[cols_to_scale])


#scaler = StandardScaler()
#data[['external_links', 'internal_links','age','total_life']] = scaler.fit_transform(data[['external_links', 'internal_links','age','total_life']])

X = pd.concat([data[['external_links', 'internal_links','age','total_life','has_Social','Same_RH_Country','Privacy','cheap_TLD','dom_has_hyphen','dom_has_digits','top1m','R_Country','H_Country','TLD']]], axis=1)


# Find the pearson correlations matrix
corr = X.corr(method = 'pearson')
print(corr)

plt.figure(figsize=(10,8), dpi =100)
sns.heatmap(corr,annot=True,fmt=".2f", linewidth=.5)
plt.show()





""" # Spearmans Correlation Analysis

data = pd.read_csv('finalDataset-FixedMissingValues-pandasT - Copy.csv')



X = pd.concat([data[['has_Social','Same_RH_Country','Privacy','cheap_TLD','dom_has_hyphen','dom_has_digits','top1m']]], axis=1)


# Find the pearson correlations matrix
corr = X.corr(method = 'spearman')
print(corr)

plt.figure(figsize=(10,8), dpi =100)
sns.heatmap(corr,annot=True,fmt=".2f", linewidth=.5)
plt.show() """





""" binary_target = 'label'


scaler = StandardScaler()
data[['external_links', 'internal_links', 'age', 'total_life']] = scaler.fit_transform(
    data[['external_links', 'internal_links', 'age', 'total_life']]
)
 """

""" # Select features including the binary target for point-biserial correlations
features = [
    'external_links', 'internal_links', 'age', 'total_life', binary_target
]
#features = [
#    'has_Social', 'Same_RH_Country', 'Privacy', 'cheap_TLD',
#    'dom_has_hyphen', 'dom_has_digits', binary_target
#]       # add top1m


# Create a new DataFrame with selected features
X = data[features]

# Calculate point-biserial correlations
correlations = {}
for feature in features[:-1]:  # Exclude the target variable
    point_biserial_corr, p_value = pointbiserialr(X[feature], X[binary_target])
    correlations[feature] = point_biserial_corr

# Convert the correlations dictionary to a DataFrame for easier visualization
correlation_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Point-Biserial Correlation'])

# Plot the results
plt.figure(figsize=(10, 8), dpi=100)
sns.barplot(x='Point-Biserial Correlation', y='Feature', data=correlation_df, orient='h')
plt.title('Point-Biserial Correlations with Binary Target')
plt.show() 
 """




""" 
data = pd.read_csv('finalDataset-FixedMissingValues-pandasT - Copy.csv')

scaler = StandardScaler()
data[['external_links', 'internal_links', 'age', 'total_life']] = scaler.fit_transform(
    data[['external_links', 'internal_links', 'age', 'total_life']]
)

# Assuming 'label' is your binary target variable
binary_target = 'label'

features = [
    'R_Country',
    'H_Country', 'TLD', binary_target
]

# Create a new DataFrame with selected features
X = data[features]


# Perform chi-squared tests for categorical features
chi2_p_values = {}
for feature in ['R_Country', 'H_Country', 'TLD']:
    contingency_table = pd.crosstab(X[feature], X[binary_target])
    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
    chi2_p_values[feature] = p_value

# Convert the p-values dictionary to a DataFrame for easier visualization
chi2_p_values_df = pd.DataFrame(list(chi2_p_values.items()), columns=['Feature', 'Chi-Squared P-Value'])

# Plot the results
plt.figure(figsize=(10, 8), dpi=100)
sns.barplot(x='Chi-Squared P-Value', y='Feature', data=chi2_p_values_df)
plt.title('Chi-Squared Test P-Values for Categorical Features')
plt.xlabel('P-Value')
plt.show() """