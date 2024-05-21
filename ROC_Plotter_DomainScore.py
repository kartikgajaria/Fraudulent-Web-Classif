import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
#df = pd.read_csv('DomainScore-TPR-FPR.csv')
df = pd.read_csv('DomainScore-LegitProb-TPR-FPR.csv')

""" #excludes categorical features
x_IG = df['FPR_IG']
y_IG = df['TPR_IG']
x_RF = df['FPR_RF']
y_RF = df['TPR_RF'] """

#includes Categorical features
x_IG = df['FPR_IG_C']
y_IG = df['TPR_IG_C']
x_RF = df['FPR_RF_C']
y_RF = df['TPR_RF_C']



x_sorted_IG = np.array(sorted(x_IG))
y_sorted_IG = y_IG[x_IG.argsort()]

x_sorted_RF = np.array(sorted(x_RF))
y_sorted_RF = y_RF[x_RF.argsort()]

auc_IG = simps(y_sorted_IG, x_sorted_IG)
auc_RF = simps(y_sorted_RF, x_sorted_RF)

plt.plot(df['FPR_IG_C'], df['TPR_IG_C'], label='via Mutual Information Weights (AUC = %0.3f)' % auc_IG)
plt.plot(df['FPR_RF_C'], df['TPR_RF_C'], label='via Feature Importance Weights (AUC = %0.3f)' % auc_RF)

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
#plt.title('Domain Scoring system')
plt.legend()
plt.show()


