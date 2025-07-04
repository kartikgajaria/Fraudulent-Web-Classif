import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc, make_scorer
)

# Load dataset
#data = pd.read_csv('primary_dataset.csv')   #use this to run on primary dataset.
data = pd.read_csv('augmented_dataset.csv')  #use this to run on augmented dataset.

# Feature selection
X = data[['external_links', 'internal_links', 'age', 'total_life', 'has_Social',
          'Same_RH_Country', 'Privacy', 'cheap_TLD', 'dom_has_hyphen',
          'dom_has_digits', 'top1m', 'R_Country', 'H_Country', 'TLD']].copy()
y = data['label']

# Encode categorical features
le = LabelEncoder()
for col in ['R_Country', 'H_Country', 'TLD']:
    X[col] = le.fit_transform(X[col])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Models without hyperparameter tuning.
""" models = {                        
    "SVM": SVC(kernel='linear', probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
    "Naive Bayes": GaussianNB()
} """


# Models with hyperparameter tuning.
models = {
    "SVM": SVC(C=10, gamma=0.01, kernel='rbf', probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=10, min_samples_split=10, max_features='sqrt', bootstrap=True, random_state=42),
    "Logistic Regression": LogisticRegression(C=0.1, class_weight=None, penalty='l2', solver='lbfgs', max_iter=300, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), alpha=0.0001, learning_rate='constant', activation='relu', solver='adam', early_stopping=True, n_iter_no_change=10, validation_fraction=0.1, max_iter=500, random_state=42),
    "Naive Bayes": GaussianNB()    
}
# Added Stacking classifier
base_learners = list(models.items())
meta_classifier = LogisticRegression(max_iter=1000, random_state=42)
models["Stacking"] = StackingClassifier(estimators=base_learners, final_estimator=meta_classifier, cv=5, stack_method='predict_proba')

# Scoring metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'roc_auc': make_scorer(roc_auc_score)
}

# Stratified K-Fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


print("\n=== Classifier Performance (5-Fold Cross-Validation) ===")
all_results = []

for name, model in models.items():
    scores = cross_validate(model, X_scaled, y, cv=cv, scoring=scoring, return_train_score=False)
    print(f"\n{name}:")
    for metric in scoring.keys():
        mean = scores[f'test_{metric}'].mean()
        std = scores[f'test_{metric}'].std()
        print(f"  {metric.capitalize():<9}: {mean:.4f} ± {std:.4f}")
    for fold in range(cv.get_n_splits()):
        all_results.append({
            'Model': name,
            'Accuracy': scores['test_accuracy'][fold],
            'Precision': scores['test_precision'][fold],
            'Recall': scores['test_recall'][fold],
            'F1': scores['test_f1'][fold],
            'AUC': scores['test_roc_auc'][fold]
        })




""" # === BOX PLOT ===
df_results = pd.DataFrame(all_results)
df_melted = df_results.melt(id_vars='Model', var_name='Metric', value_name='Score')

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_melted, x='Metric', y='Score', hue='Model')
plt.title('Classifier Performance Comparison (5-Fold CV)')
plt.ylabel('Score')
plt.xlabel('Performance Metric')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show() """

""" # === PLOT ROC CURVES ===
plt.figure(figsize=(10, 8))
for name, model in models.items():
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train_idx, test_idx in cv.split(X_scaled, y):
        model.fit(X_scaled[train_idx], y[train_idx])
        y_proba = model.predict_proba(X_scaled[test_idx])[:, 1]
        fpr, tpr, _ = roc_curve(y[test_idx], y_proba)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr,
             label=f'{name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve - 5-Fold Cross-Validation')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show() """




""" Plot Learning Curves
from sklearn.model_selection import learning_curve, StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import os

train_sizes = [600, 650, 700, 750, 770]
train_sizes, train_scores, val_scores = learning_curve(
    models['SVM'],
    X_scaled,
    y,
    train_sizes=train_sizes,
    cv=cv,
    scoring='accuracy',
    error_score='raise'
)

# Remove any NaN values
train_scores_mean = train_scores.mean(axis=1)
val_scores_mean = val_scores.mean(axis=1)

plt.figure(figsize=(5, 5))
plt.plot(train_sizes, train_scores_mean, 'o-', label='Train Score')
plt.plot(train_sizes, val_scores_mean, 'o-', label='Validation Score')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve - Logistic Regression')
plt.legend()
plt.grid(True)
plt.show()

print(pd.DataFrame(train_scores).isna().sum(axis=1))  # See how many folds failed per train size """




""" # Save Learning Curve Plots
output_dir = "C:\\Users\\kgaja\\OneDrive\\Pictures\\Paper\\Learning Curves\\Before"
os.makedirs(output_dir, exist_ok=True)

# Loop through each model
for model_name, model in models.items():
    print(f"Processing: {model_name}")
    
    try:
        train_sizes_eff, train_scores, val_scores = learning_curve(
            model,
            X_scaled,
            y,
            train_sizes=train_sizes,
            cv=cv,
            scoring='accuracy',
            error_score='raise'
        )
        
        # Compute mean scores
        train_scores_mean = np.mean(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)

        # Plot
        plt.figure(figsize=(5, 5))
        plt.plot(train_sizes_eff, train_scores_mean, 'o-', label='Train Score')
        plt.plot(train_sizes_eff, val_scores_mean, 'o-', label='Validation Score')
        plt.xlabel('Training Size')
        plt.ylabel('Accuracy')
        #plt.title(f'Learning Curve - {model_name}')
        plt.legend()
        plt.grid(True)

        # Save plot
        save_path = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_LC_before.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    except Exception as e:
        print(f"Failed for {model_name}: {e}")

 """





""" Hyperparameter tuning via gridsearch MLP
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (64, 32)],
    'alpha': [1e-4, 1e-3, 1e-2],
    'learning_rate': ['constant', 'adaptive'],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    MLPClassifier(max_iter=300, early_stopping=True, random_state=42),
    param_grid,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=2  # Optional: shows live output
)

grid.fit(X_scaled, y)

print("Best params:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)
 """



""" Hyperparameter tuning via gridsearch LR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs'],  # change to 'liblinear' or 'saga' if using 'l1' or 'elasticnet'
    'class_weight': [None, 'balanced'],
    'max_iter': [300]
}

grid = GridSearchCV(
    LogisticRegression(),
    param_grid,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1
)

grid.fit(X_scaled, y)
print("Best parameters:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)
 """


""" Hyperparameter tuning via gridsearch SVM
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.1, 1, 10],  # 'scale' is sklearn default
    'kernel': ['rbf']  # RBF works well in most non-linear cases
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    SVC(),
    param_grid,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_scaled, y)

print("Best parameters:", grid.best_params_)
print("Best CV score:", grid.best_score_) """



""" Plotting Confusion Matrix
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Define your target directory
save_dir = r"C:\\Users\\kgaja\\OneDrive\\Pictures\\Paper\\Aug\\Tuned"
os.makedirs(save_dir, exist_ok=True)

for name, model in models.items():
    print(f"\n{name}:")
    fold_idx = 0
    for train_idx, test_idx in cv.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute and plot confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        fig, ax = plt.subplots(figsize=(3, 3))
        #disp.plot(ax=ax, cmap='Blues', values_format='d')
        disp.plot(ax=ax, text_kw={"fontsize": 18}, cmap='Blues')
         # Increase axis label font sizes
        ax.set_xlabel('Predicted label', fontsize=14)
        ax.set_ylabel('True label', fontsize=14)
        ax.tick_params(axis='both', labelsize=13)  # Adjust tick label font size
        #plt.title(f"{name} - Fold {fold_idx}")
        #plt.tight_layout()

        # Save the plot
        filename = f"{name}_Fold{fold_idx}_CM.png"
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

        fold_idx += 1
 """