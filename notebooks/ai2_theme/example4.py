import matplotlib.pyplot as plt
import numpy as np

# Example data
categories = ["A", "B", "C", "D"]
group1 = np.array([5, 3, 4, 7])
group2 = np.array([2, 4, 6, 1])
group3 = np.array([3, 2, 5, 2])

# Position of bars on y-axis
y_pos = np.arange(len(categories))

# Create horizontal stacked bars
plt.barh(y_pos, group1, color="skyblue", label="Group 1")
plt.barh(y_pos, group2, left=group1, color="lightgreen", label="Group 2")
plt.barh(y_pos, group3, left=group1 + group2, color="salmon", label="Group 3")

# Add labels and legend
plt.yticks(y_pos, categories)
plt.xlabel("Values")
plt.title("Horizontal Stacked Bar Chart Example")
plt.legend()

plt.tight_layout()
plt.show()
