import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the confusion matrix
conf_matrix = np.array([    [20,  0,  0,  1],    [ 0, 19,  2,  0],    [ 0,  3, 18,  0],    [ 0,  0,  0, 21]])
#conf_matrix = np.array([[65, 1, 0, 1], [9, 45, 4, 9], [0, 6, 30, 31], [3, 15, 12, 37]])
#conf_matrix = np.array([[12, 2, 0, 0, 0, 0, 0], [3, 9, 2, 0, 0, 0, 0], [0, 2, 8, 2, 0, 0, 2], [0, 1, 1, 8, 2, 0, 2], [0, 0, 2, 4, 8, 0, 0], [0, 1, 2, 2, 8, 1, 0], [0, 0, 0, 0, 0, 0, 14]])
#conf_matrix = np.array([[17, 0, 3, 3, 6, 1], [13, 2, 9, 0, 5, 1], [8, 0, 18, 0, 3, 1], [6, 0, 0, 4, 6, 14], [5, 0, 2, 1, 21, 1], [2, 0, 0, 0, 0, 28]])
#conf_matrix = np.array([[84, 65, 7, 5], [11, 111, 34, 5], [3, 45, 92, 21], [0, 0, 9, 152]])

# Define the classes (you may want to customize these names)
#classes = [1, 2, 3, 4]
classes = ['Axx', 'Csi','Eac','Hsx']
#classes = ['Axx', 'Bxi', 'Cai', 'Cso']
#classes = ['A', 'B', 'C', 'D', 'E', 'F', 'H']
#classes = ['a', 'h', 'k', 'r', 's', 'x']
#classes = ['c', 'i', 'o', 'x']

y_actu = []
y_pred = []

# Generate actual and predicted values from the confusion matrix
for i in range(len(conf_matrix)):
    for j in range(len(conf_matrix[i])):
        y_actu.extend([i] * conf_matrix[i][j])
        y_pred.extend([j] * conf_matrix[i][j])

# Create a DataFrame for confusion matrix
df_confusion = pd.crosstab(y_actu, y_pred)

def plot_confusion_matrix(cm, classes, cmap=plt.cm.Greens):
    plt.figure(figsize=(10, 8))
    plt.matshow(cm, cmap=cmap)
    plt.gca().xaxis.set_ticks_position('bottom')

    plt.colorbar()
    tick_marks = np.arange(1, len(classes) + 1, 3)  # Adjust tick positions
    '''    
    cbar = plt.colorbar()  # Ulož colorbar do proměnné

    tick_marks = np.arange(0, 22, 3)  # Nastav ticky každé 3 (0,3,6,...)
    cbar.set_ticks(tick_marks)
    cbar.set_ticklabels(tick_marks)'''

    plt.xticks(range(len(classes)), classes, fontsize=12)  # Adjust tick labels
    plt.yticks(range(len(classes)), classes, fontsize=12)  # Adjust tick labels

    plt.subplots_adjust(top=0.85, bottom=0.2)  # Fix the top and bottom margins
    plt.xlabel('Predicted Class', fontsize=18)  # Set x-axis label
    plt.ylabel('True Class', fontsize=18)  # Set y-axis label
    plt.title('Confusion Matrix 1', fontsize=22, pad=20) # Customize title


    plt.savefig(f"EUCYS/Confusion_Matrix_{classes}.svg")  # Save the figure
    plt.show()

# Plot the confusion matrix
plot_confusion_matrix(df_confusion, classes, cmap=plt.cm.Greens)
