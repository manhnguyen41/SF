
def create_heatmap(data_2d, filename="heatmap.png", title="Heatmap", x_label="X-axis", y_label="Y-axis", cmap="viridis"):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    data = np.array(data_2d)
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=False, fmt='.2f', cmap=cmap, cbar=True, square=False)
    plt.title(title, pad=20)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # Save the figure
    plt.savefig(f"fig/{filename}", dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to free memory
    
    print(f"Heatmap saved as {filename}")