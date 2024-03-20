import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

conf_matrix = np.array([[12, 0, 0, 0],
                        [1, 10, 1,0],
                        [3, 3, 3,3],
                        [2,6,4,0]])

y_actu = []
y_pred = []

for i in range(len(conf_matrix)):
    for j in range(len(conf_matrix[i])):
        y_actu.extend([i] * conf_matrix[i][j])
        y_pred.extend([j] * conf_matrix[i][j])

df_confusion = pd.crosstab(y_actu, y_pred)

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    plt.colorbar()
    tick_marks = np.arange(1, len(df_confusion.columns) + 1)  # Adjust tick positions
    plt.xticks(tick_marks - 1, tick_marks, fontsize=12)  # Adjust tick labels
    plt.yticks(tick_marks - 1, tick_marks, fontsize=12)  # Adjust tick labels
    plt.xlabel('Správná třída', fontsize=18) # Customize y-axis label
    plt.ylabel('Predikovaná třída', fontsize=18) # Customize x-axis label
    plt.title('Konfuzní matice', fontsize=22) # Customize title

plt.figure(figsize=(8, 6))
plot_confusion_matrix(df_confusion, title='Customized Confusion Matrix', cmap=plt.cm.Greens) # Change title and color scheme
plt.show()
