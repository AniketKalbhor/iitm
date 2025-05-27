import numpy as np
np.set_printoptions(threshold=np.inf)  # This will show the full array
data = np.load(r"C:\Users\karan\Downloads\ai\pooled_features\2024-06-17_pooled_features.npy")
print(data)

data = np.load(r"combined_features/2024-06-17_all_features.npy", allow_pickle=True).item()
print(data)
