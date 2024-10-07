import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('data_for_soc.csv')

# Plot the data
plt.figure(figsize=(8, 6))

# Plot each line with customizations
plt.plot(df['1'], label='Accuracy', color='orange', linestyle='-')
plt.plot(df['2'], label='Loss Function', color='red', linestyle='-')
plt.plot(df['3'], label='Validation Accuracy', color='green', linestyle='-')
plt.plot(df['4'], label='Validation Loss Function', color='blue', linestyle='-', linewidth=3)

plt.xlabel('Epoch', fontsize=14)
plt.title('Development of Accuracy and Loss Function', fontsize=20)
plt.legend(loc='upper right')  # Adjust the location of the legend
# Set the plot range (xlim and ylim)
plt.xlim(0, len(df) - 1)  # Adjust x-axis range from 0 to the length of the DataFrame minus 1
plt.ylim(0, 1.5)  # Adjust y-axis range from 0 to 2

plt.tight_layout()  # Adjust layout to prevent overlap

# Save the plot as an image (optional)
plt.savefig('custom_graph.svg')


# Show the plot
plt.show()
