import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the confusion matrix
conf_matrix = np.array([[12, 0, 0, 0],
                        [1, 10, 1, 0],
                        [3, 3, 3, 3],
                        [2, 6, 4, 0]])

y_actu = []
y_pred = []

# Generate actual and predicted values from the confusion matrix
for i in range(len(conf_matrix)):
    for j in range(len(conf_matrix[i])):
        y_actu.extend([i] * conf_matrix[i][j])
        y_pred.extend([j] * conf_matrix[i][j])

# Create a DataFrame for confusion matrix
df_confusion = pd.crosstab(y_actu, y_pred)

# Define the classes (you may want to customize these names)
classes = [1, 2, 3, 4]

def plot_confusion_matrix(cm, classes, cmap=plt.cm.Greens):
    plt.figure(figsize=(10, 8))
    plt.matshow(cm, cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(1, len(classes) + 1)  # Adjust tick positions
    plt.xticks(range(len(classes)), classes, fontsize=12)  # Adjust tick labels
    plt.yticks(range(len(classes)), classes, fontsize=12)  # Adjust tick labels

    plt.subplots_adjust(top=0.85, bottom=0.2)  # Fix the top and bottom margins
    plt.xlabel('Predicted Class', fontsize=18)  # Set x-axis label
    plt.ylabel('True Class', fontsize=18)  # Set y-axis label
    plt.title('Confusion Matrix', fontsize=22) # Customize title


    plt.savefig(f"sample_Confusion_Matrix.svg")  # Save the figure
    plt.show()

# Plot the confusion matrix
plot_confusion_matrix(df_confusion, classes, cmap=plt.cm.Greens)
