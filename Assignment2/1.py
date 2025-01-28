import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the range of x values
x_values = np.linspace(-5, 5, 100)  # 100 evenly spaced points between -5 and 5

# Compute y values for the three equations
line1 = 2 * x_values + 1  # Equation: y = 2x + 1
line2 = 2 * x_values + 2  # Equation: y = 2x + 2
line3 = 2 * x_values + 3  # Equation: y = 2x + 3

# Plot the graphs
plt.figure(figsize=(8, 6))  # Set the figure size to 8x6 inches

# Plot each line with different styles, colors, and labels
plt.plot(x_values, line1, label=r'$y=2x+1$', color='blue', linestyle='-', linewidth=2)
plt.plot(x_values, line2, label=r'$y=2x+2$', color='green', linestyle='--', linewidth=2)
plt.plot(x_values, line3, label=r'$y=2x+3$', color='red', linestyle=':', linewidth=2)

# Add title and axis labels
plt.title('Graph of y=2x+1, y=2x+2, y=2x+3', fontsize=14)  # Title for the graph
plt.xlabel('Input (x)', fontsize=12)  # Label for x-axis
plt.ylabel('Output (y)', fontsize=12)  # Label for y-axis

# Add a legend to differentiate the lines
plt.legend()

# Enable gridlines for better readability
plt.grid(True)

# Display the graph
plt.show()
