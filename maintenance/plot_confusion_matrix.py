import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

conf_matrix = np.array([[65, 1, 0, 1],
                        [9, 45, 4, 9],
                        [0, 6, 30, 31],
                        [3, 15, 12, 37]])

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
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    plt.xlabel('Správná třída', fontsize=12) # Customize y-axis label
    plt.ylabel('Predikovaná třída', fontsize=12) # Customize x-axis label
    plt.title('Konfuzní matice', fontsize=18) # Customize title

plt.figure(figsize=(8, 6))
plot_confusion_matrix(df_confusion, title='Customized Confusion Matrix', cmap=plt.cm.Greens) # Change title and color scheme
plt.show()
