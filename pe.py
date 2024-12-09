import numpy as np
import matplotlib.pyplot as plt

def sinusoidal_positional_encoding(max_position, d_model):
    position = np.arange(max_position)[:, np.newaxis]
    # The original formula pos / 10000^(2i/d_model) is equivalent to pos * (1 / 10000^(2i/d_model)).
    # I use the below version for numerical stability
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    print(np.log(10000))
    
    pe = np.zeros((max_position, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe

max_position = 50  # Maximum sequence length
d_model = 128        # Embedding dimension

pe = sinusoidal_positional_encoding(max_position, d_model)

# plt.figure(figsize=(12, 8))
# plt.imshow(pe, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
# plt.colorbar()
# plt.title('Sinusoidal Positional Encoding')
# plt.xlabel('Dimension')
# plt.ylabel('Position')
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12, 6))
# dimensions = [0, 21]
# for d in dimensions:
#     plt.plot(pe[:, d], label=f'Dim {d}')
# plt.legend()
# plt.title('Sinusoidal Positional Encoding for Specific Dimensions')
# plt.xlabel('Position')
# plt.ylabel('Value')
# plt.tight_layout()
# plt.show()

def cosine_similarity(A, B):
    # Compute the dot product of A and B
    dot_product = np.dot(A, B)
    
    # Compute the magnitude (norm) of each vector
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    
    # Compute the cosine similarity
    similarity = dot_product / (norm_A * norm_B)
    
    return similarity

sim = np.zeros((max_position,max_position))

for i in range(0,max_position):
    for j in range(0,max_position):
        sim[i, j] = cosine_similarity(pe[i],pe[j])

# print(sim)

x = np.arange(max_position)
y = np.arange(max_position)
X, Y = np.meshgrid(x, y)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface using the cosine similarity matrix
ax.plot_surface(X, Y, sim, cmap='coolwarm')

# Add labels and title
ax.set_xlabel('X (Position)')
ax.set_ylabel('Y (Position)')
ax.set_zlabel('Cosine Similarity')
ax.set_title('Cosine Similarity 3D Plot')

# Show the plot
plt.show()