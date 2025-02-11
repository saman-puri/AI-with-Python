import numpy as np
import matplotlib.pyplot as plt

# Different sample sizes to see how results change with more rolls
sample_sizes = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

# Loop through each sample size and simulate rolling two dice
for sample_size in sample_sizes:
    # Simulate rolling two dice multiple times
    first_dice = np.random.randint(1, 7, sample_size)
    second_dice = np.random.randint(1, 7, sample_size)
    total_sum = first_dice + second_dice  # Calculate the sum of the dice
    
    # Count how often each sum appears
    counts, bin_edges = np.histogram(total_sum, bins=range(2, 14))
    
    # Create the histogram
    plt.bar(bin_edges[:-1], counts / sample_size, align='center')
    plt.xlabel('Sum of Two Dice')
    plt.ylabel('Relative Frequency')
    plt.title(f'How Often Each Sum Appears (n={sample_size})')
    plt.xticks(range(2, 13))
    
    # Add percentage labels above each bar for clarity
    for value, frequency in zip(bin_edges[:-1], counts / sample_size):
        plt.text(value, frequency, f"{frequency * 100:.1f}%", ha='center', va='bottom')
    
    plt.show()