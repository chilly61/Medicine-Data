import matplotlib.pyplot as plt
from RF import RF_Detection
import numpy as np
from LightGBM import LightGBM_Detection


def plot_result(x, scores, x_label='', x_scale='linear'):
    plt.figure(figsize=(10, 6))
    plt.plot(x, scores[0], label='Score for X')
    plt.plot(x, scores[1], label='Score for Z')
    plt.xlabel(x_label)
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs {x_label}')
    plt.legend()
    plt.xscale(x_scale)
    plt.grid(True)
    plt.show()


original_data_path = './样例数据.xlsx'
RFD = RF_Detection()
"""
## Hyper parameter Graph on n_estimators
n_estimators = np.array([10**i for i in range(5)])
scores = np.zeros((2, len(n_estimators)))
for i, n in enumerate(n_estimators):
    RFD.fit(original_data_path, n_estimators=n)
    scores[0, i] = RFD.score('X')
    scores[1, i] = RFD.score('Z') """
# plot_result(n_estimators, scores, x_label = 'Number of Estimators', x_scale = 'log')

# The best n_estimator seems to be 10 in the prev results
""" max_depths = np.array([None] + [2**i for i in range(10)])
scores = np.zeros((2, len(max_depths)))
for i, d in enumerate(max_depths):
    ## set n_estimators = 10
    RFD.fit(original_data_path, n_estimators=10, max_depth=d)
    scores[0, i] = RFD.score('X')
    scores[1, i] = RFD.score('Z')

max_depths_for_plot = ['Unlimited' if d is None else d for d in max_depths]
plot_result(max_depths_for_plot, scores, x_label='Maximum Depths', x_scale='linear') """

""" max_features = ['sqrt', 'log2', None]
scores = np.zeros((2, len(max_features)))
for i, f in enumerate(max_features):
    # Set n_estimators = 10 and vary max_features
    RFD.fit(original_data_path, n_estimators=10, max_features=f)
    scores[0, i] = RFD.score('X')
    scores[1, i] = RFD.score('Z')
max_features_labels  = ['None' if f is None else f for f in max_features]
plt.figure(figsize=(10, 6))
plt.plot(max_features_labels, scores[0, :], marker='o', label='Score X')
plt.plot(max_features_labels, scores[1, :], marker='x', label='Score Z')

plt.xlabel('Max Features')
plt.ylabel('Scores')
plt.title('Scores by Max Features')
plt.xticks(max_features_labels)  # Ensure only specified labels are used
plt.legend()
plt.show() """

# Test Statistical Function
# RFD.fit(original_data_path, n_estimators=10, max_features='log2')
# RFD.statistical_analysis(save = True)

LGBM = LightGBM_Detection()
LGBM.fit(original_data_path)
LGBM.statistical_analysis(save=True)
