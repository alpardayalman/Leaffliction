import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


model = models.resnet18(pretrained=True)
model = model.eval()  # Set to evaluation mode

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_image_embedding(image_path):
    # Open and transform the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model(image)

    embedding = embedding.squeeze().numpy()
    return embedding / np.linalg.norm(embedding)


image_folder = "leaves/images"
image_paths = []
embeddings = []
folder_names = []


for root, dirs, files in os.walk(image_folder):
    for file in tqdm(files, desc="Processing images", ncols=100):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(root, file)
            folder_name = os.path.basename(root)
            embedding = get_image_embedding(image_path)
            image_paths.append(image_path)
            embeddings.append(embedding)
            folder_names.append(folder_name)

embeddings = np.array(embeddings)

df = pd.DataFrame({
    'Folder Name': folder_names,
    'Image Path': image_paths,
    'Embedding': [emb.tolist() for emb in embeddings]
})

df.to_csv('image_embeddings.csv', index=False)
