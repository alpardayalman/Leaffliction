import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity
import os

model = models.resnet18(pretrained=True)
model = model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model(image)

    embedding = embedding.squeeze().numpy()
    return embedding / np.linalg.norm(embedding)


df = pd.read_csv('image_embeddings.csv')
image_paths = df['Image Path'].tolist()
embeddings = np.array(
    df['Embedding'].apply(lambda x: np.fromstring(x[1:-1], sep=',')).tolist())


def search_similar_images(query_embedding, embeddings, image_paths, k=5):
    similarities = cosine_similarity([query_embedding], embeddings)
    top_k_indices = np.argsort(similarities[0])[::-1][:k]
    top_k_distances = similarities[0][top_k_indices]
    top_k_image_paths = [image_paths[i] for i in top_k_indices]

    return top_k_image_paths, top_k_distances


st.title('Image Retrieval with Brute Force Search')
st.write("Upload an image to search for similar images.")

uploaded_image = st.file_uploader(
    "Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    query_image = Image.open(uploaded_image).convert("RGB")
    st.image(query_image, caption="Uploaded Image", use_container_width=True)

    query_embedding = get_image_embedding(uploaded_image)

    k = 6
    top_k_image_paths, kd = search_similar_images(
        query_embedding, embeddings, image_paths, k)

    st.write(f"Top {k} similar images:")

    for i in range(1, k):
        image_path = top_k_image_paths[i]
        fn = os.path.basename(os.path.dirname(image_path))
        if image_path != uploaded_image.name:
            a = f"Folder: {fn} | Similar Image {i+1} (Distance: {kd[i]:.2f})"
            st.image(image_path,
                     caption=a,
                     use_container_width=True)
