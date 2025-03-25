import faiss
import numpy as np
import pandas as pd

# Assuming you have the image embeddings in a DataFrame (e.g., 'image_embeddings.csv')
df = pd.read_csv('image_embeddings.csv')  # Replace with your CSV file path
image_paths = df['Image Path'].tolist()
embeddings = np.array(df['Embedding'].apply(lambda x: np.fromstring(x[1:-1], sep=',')).tolist())

# Build the FAISS HNSW index
dim = embeddings.shape[1]  # The dimension of the embeddings (e.g., 512 for ResNet18)
index = faiss.IndexHNSWFlat(dim, 32)  # 32 is the number of neighbors in the HNSW graph

# Add the embeddings to the index
index.add(embeddings.astype(np.float32))

# Save the index to a file
faiss.write_index(index, "image_index.index")

# Optionally, print to confirm
print(f"FAISS index has been saved to 'image_index.index'")
