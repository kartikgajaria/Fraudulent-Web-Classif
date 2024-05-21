import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import make_classification


data = pd.read_csv('primary_dataset.csv')   # Use this for <------- PRIMARY
#data = pd.read_csv('augmented_dataset.csv')        # Use this for <------- AUGMENTED

pastData = pd.read_csv('historical_dataset.csv')


X = pd.concat([data[['external_links', 'internal_links','age','total_life','has_Social','Same_RH_Country','Privacy','cheap_TLD','dom_has_hyphen','dom_has_digits','top1m','R_Country','H_Country','TLD']]], axis=1)

y = pd.concat([data[['label']]],axis=1)

z = pd.concat([pastData[['external_links', 'internal_links','age','total_life','has_Social','Same_RH_Country','Privacy','cheap_TLD','dom_has_hyphen','dom_has_digits','top1m','R_Country','H_Country','TLD']]], axis=1)
l = pd.concat([pastData[['label']]],axis=1)


print("############################################################################")
print("#############              SVM Model            ############################")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

model = SVC(kernel='linear',probability=True, random_state=42)  
model.fit(X_train, y_train.values.ravel())
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print('Classification Report:')
print(classification_report(y_test, y_pred, digits=2))

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))



# Calculate ROC curve for PRIMARY
y_pred_proba_SVM = model.predict_proba(X_test)[:, 1]
fpr_SVM, tpr_SVM, thresholds_SVM = roc_curve(y_test, y_pred_proba_SVM, pos_label=1) 
roc_auc_SVM = auc(fpr_SVM, tpr_SVM)



""" # Uncomment to plot the confusion matrix for SVM on PRIMARY
conf_matrix = confusion_matrix(y_test, y_pred,labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=model.classes_)
disp.plot()
print("SVM CF")
plt.show() """



""" # Uncomment to see results of testing on HISTORICAL
print("#########  Testing PAST DATA  ##############")

l_pred = model.predict(z)

accuracy = accuracy_score(l, l_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print('Classification Report:')
print(classification_report(l, l_pred,zero_division=0, digits=2))

print('Confusion Matrix:')
print(confusion_matrix(l, l_pred))"""



""" # Uncomment to plot the confusion matrix of SVM for HISTORICAL
conf_matrix = confusion_matrix(l, l_pred,labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=model.classes_)
disp.plot()
print("SVM PAST CF")
plt.show() """




print("@@@@@@@@@@@@@@@@@@        RANDOM FOREST       @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

model = RandomForestClassifier(n_estimators=100, random_state=42)  
model.fit(X_train, y_train.values.ravel())

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print('Classification Report:')
print(classification_report(y_test, y_pred, digits=2))

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))


# Calculate ROC curve for RF on PRIMARY

y_pred_proba_RF = model.predict_proba(X_test)[:, 1]
fpr_RF, tpr_RF, thresholds_RF = roc_curve(y_test, y_pred_proba_RF, pos_label=1) 
roc_auc_RF = auc(fpr_RF, tpr_RF)

""" # Uncomment to plot the confusion matrix for RF on PRIMARY
conf_matrix = confusion_matrix(y_test, y_pred,labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=model.classes_)
disp.plot()
print("RF CF")
plt.show() """


""" # Uncomment for RF's feature importance
global_importances = pd.Series(model.feature_importances_, index=X_train.columns)
global_importances.sort_values(ascending=True, inplace=True)
plt.figure(figsize=(12.2, 8))
global_importances.plot.barh(color='green')
plt.xlabel("$\\mathbf{Importance}$", fontsize=12, fontweight='bold')
plt.ylabel("Feature")
plt.title("Feature Importance")
# Making feature labels bold with LaTeX
labels = ['$\\mathbf{' + label.replace('_', '\_') + '}$' for label in global_importances.index]
plt.yticks(range(len(global_importances.index)), labels)
# Set x-axis tick labels to bold
plt.xticks(fontweight='bold')
plt.show()
print(global_importances)
global_importances.to_csv("RF's Feature Importance Weights.csv")
 """



""" # Uncomment to test trained RF on HISTORICAL
print("#########  Testing PAST DATA  ##############")
l_pred = model.predict(z)
accuracy = accuracy_score(l, l_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print('Classification Report:')
print(classification_report(l, l_pred,zero_division=0, digits=2))

print('Confusion Matrix:')
print(confusion_matrix(l, l_pred))"""


""" # Uncomment to plot the confusion matrix of RF on HISTORICAL

conf_matrix = confusion_matrix(l, l_pred,labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=model.classes_)
disp.plot()
print("RF PAST CF")
plt.show() """





print("#############              Log Regression Model            ############################")

model = LogisticRegression(random_state=42, max_iter=1000) 
model.fit(X_train, y_train.values.ravel())
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print('Classification Report:')
print(classification_report(y_test, y_pred, digits=2))

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))


# Calculate ROC curve for LR on PRIMARY

y_pred_proba_LR = model.predict_proba(X_test)[:, 1]
fpr_LR, tpr_LR, thresholds_LR = roc_curve(y_test, y_pred_proba_LR, pos_label=1) 
roc_auc_LR = auc(fpr_LR, tpr_LR)




""" # Uncomment to plot the confusion matrix for LR on PRIMARY
conf_matrix = confusion_matrix(y_test, y_pred,labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=model.classes_)
disp.plot()
print("LR CF")
plt.show() """


""" # Uncomment to check trained LR on HISTORICAL
print("#########  Testing PAST DATA  ##############")
l_pred = model.predict(z)

accuracy = accuracy_score(l, l_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print('Classification Report:')
print(classification_report(l, l_pred,zero_division=0, digits=2))

print('Confusion Matrix:')
print(confusion_matrix(l, l_pred))"""



""" # Uncomment to plot the confusion matrix for LR on HISTORICAL
conf_matrix = confusion_matrix(l, l_pred,labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=model.classes_)
disp.plot()
print("LR PAST CF")
plt.show() """



print("#############              MLP Model            ############################")


mlp = MLPClassifier(hidden_layer_sizes=(64, 32),      
                    max_iter=1000, random_state=42)
mlp.fit(X_train, y_train.values.ravel())
y_pred = mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

class_report = classification_report(y_test, y_pred, digits=2)
print("Classification Report:\n", class_report)

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))


# Calculate ROC curve for MLP on PRIMARY

y_pred_proba_MLP = mlp.predict_proba(X_test)[:, 1]
fpr_MLP, tpr_MLP, thresholds_MLP = roc_curve(y_test, y_pred_proba_MLP, pos_label=1) 
roc_auc_MLP = auc(fpr_MLP, tpr_MLP)




""" # Uncomment to plot the confusion matrix for MLP on PRIMARY
conf_matrix = confusion_matrix(y_test, y_pred,labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=model.classes_)
disp.plot()
print("MLP CF")
plt.show() """



""" print("#########  Testing PAST DATA  ##############")

l_pred = mlp.predict(z)

# Calculate the accuracy of the model
accuracy = accuracy_score(l, l_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate a classification report
class_report = classification_report(l, l_pred,zero_division=0, digits=2)
print("Classification Report:\n", class_report)

print('Confusion Matrix:')
print(confusion_matrix(l, l_pred))"""


""" # Uncomment to plot the confusion matrix for MLP on HISTORICAL
conf_matrix = confusion_matrix(l, l_pred,labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=model.classes_)
disp.plot()
print("MLP PAST CF")
plt.show() """


""" #--------------------- Uncomment to get Mutual Information weights----------------


feature_columns = ['external_links', 'internal_links','age','total_life','has_Social','Same_RH_Country','Privacy','cheap_TLD','dom_has_hyphen','dom_has_digits','top1m','R_Country','H_Country','TLD']
features = pd.DataFrame(X, columns=feature_columns)

# Get the mutual information coefficients and convert them to a data frame
coeff_df = pd.Series(mutual_info_classif(X, y.values.ravel(), random_state=42), index=feature_columns)
coeff_df.sort_values(ascending=True, inplace=True)

# Plotting with LaTeX rendering for bold labels and handling underscores
plt.figure(figsize=(12.2, 8))
coeff_df.plot(kind='barh', color='green')
plt.xlabel("$\\mathbf{Importance}$", fontsize=12, fontweight='bold')
plt.ylabel("Feature")
plt.title("Mutual Information")

# Handling underscores in labels for LaTeX rendering
labels = ['$\\mathbf{' + label.replace('_', '\_') + '}$' for label in coeff_df.index]  # Note the use of \\ to escape backslash
plt.yticks(range(len(coeff_df.index)), labels)
print(coeff_df)
plt.xticks(fontweight='bold')
plt.show()
#coeff_df.to_csv("Mutual Information Weights.csv") """




print("##########################-----Naive Bayes------########################")
model = GaussianNB()
model.fit(X_train, y_train.values.ravel())

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

class_report = classification_report(y_test, y_pred, digits=2)
print("Classification Report:\n", class_report)

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))


# Calculate ROC curve for NB on PRIMARY
y_pred_proba_NB = model.predict_proba(X_test)[:, 1]
fpr_NB, tpr_NB, thresholds_NB = roc_curve(y_test, y_pred_proba_NB, pos_label=1) 
roc_auc_NB = auc(fpr_NB, tpr_NB)


""" # Uncomment to plot the confusion matrix for NB on PRIMARY
conf_matrix = confusion_matrix(y_test, y_pred,labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=model.classes_)
disp.plot()
print("NB CF")
plt.show() """


""" # Uncomment to test trained NB on HISTORICAL
print("#########  Testing PAST DATA  ##############")

l_pred = model.predict(z)

accuracy = accuracy_score(l, l_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print('Classification Report:')
print(classification_report(l, l_pred,zero_division=0, digits=2))

print('Confusion Matrix:')
print(confusion_matrix(l, l_pred))"""



""" # Uncomment to plot the confusion matrix for NB on HISTORICAL
conf_matrix = confusion_matrix(l, l_pred,labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=model.classes_)
disp.plot()
print("NB PAST CF")
plt.show() """



""" # Plot the ROC curve for all classifiers
plt.figure()  
plt.plot(fpr_SVM, tpr_SVM, label='SVM (AUC = %0.3f)' % roc_auc_SVM)
plt.plot(fpr_RF, tpr_RF, label='RF (AUC = %0.3f)' % roc_auc_RF)
plt.plot(fpr_LR, tpr_LR, label='LR (AUC = %0.3f)' % roc_auc_LR)
plt.plot(fpr_MLP, tpr_MLP, label='MLP (AUC = %0.3f)' % roc_auc_MLP)
plt.plot(fpr_NB, tpr_NB, label='NB (AUC = %0.3f)' % roc_auc_NB)
#plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Fraudulent E-Commerce website detection')
plt.legend()
plt.show()
""" 



""" # Uncomment to write fpr, tpr values for all classifiers to CSV if manual plotting of ROC is needed
# SVM
df_SVM = pd.DataFrame({
    'fpr_SVM': fpr_SVM,
    'tpr_SVM': tpr_SVM
})
df_SVM.to_csv('roc_data_SVM_PRIM_classif.csv', index=False)
     

# RF
df_RF = pd.DataFrame({
    'fpr_RF': fpr_RF,
    'tpr_RF': tpr_RF
})
df_RF.to_csv('roc_data_RF_PRIM_classif.csv', index=False)

# LR
df_LR = pd.DataFrame({
    'fpr_LR': fpr_LR,
    'tpr_LR': tpr_LR
})
df_LR.to_csv('roc_data_LR_PRIM_classif.csv', index=False)


# MLP
df_MLP = pd.DataFrame({
    'fpr_MLP': fpr_MLP,
    'tpr_MLP': tpr_MLP
})
df_MLP.to_csv('roc_data_MLP_PRIM_classif.csv', index=False)


# NB
df_NB = pd.DataFrame({
    'fpr_NB': fpr_NB,
    'tpr_NB': tpr_NB
})
df_NB.to_csv('roc_data_NB_PRIM_classif.csv', index=False) """


