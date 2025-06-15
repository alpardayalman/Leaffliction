import streamlit as st
import json
from PIL import Image

import torch

from src.model.network import Net
from src.model.dataset import transform_scheme1

import matplotlib.pyplot as plt


device = "cpu"


# ---------- Model & Encoder Loader ----------
@st.cache_resource
def load_model(model_file):
    model = Net()
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    return model


@st.cache_resource
def load_encoder(encoder_file):
    return json.load(encoder_file)


# ---------- Hooks for Feature Maps ----------
def get_activation(name, registry):
    def hook(model, input, output):
        if name in registry:
            key = name + "2"
        else:
            key = name
        registry[key] = output.detach()
    return hook


def main():
    model = None
    encoder = None
    activation_registry = {}

    # ---------- App Title ----------
    st.title("CNN Image Classifier & Feature Map Visualizer")

    # ---------- File Uploads ----------
    st.sidebar.header("Load Model and Label Encoder")

    model_file = st.sidebar.file_uploader("Upload PyTorch Model (.pth)",
                                          type=["pth"])
    encoder_file = st.sidebar.file_uploader("Upload Label Encoder (.json)",
                                            type=["json"])

    if model_file:
        model = load_model(model_file)
    if encoder_file:
        encoder = load_encoder(encoder_file)

    if model:
        # Register hooks (update layer names as needed)
        model.conv1.register_forward_hook(get_activation('conv1',
                                                         activation_registry))
        model.conv2.register_forward_hook(get_activation('conv2',
                                                         activation_registry))
        model.pool.register_forward_hook(get_activation('pool',
                                                        activation_registry))

    transform = transform_scheme1()

    # ---------- Image Upload ----------
    uploaded_file = st.file_uploader("Upload an image",
                                     type=["jpg", "jpeg", "png"])
    if uploaded_file:
        activation_registry.clear()

    # ---------- Processing ----------
    if uploaded_file and model:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        input_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_idx = int(output.argmax(dim=1).item())
            label = encoder[predicted_idx] if encoder else str(predicted_idx)

            st.markdown(f"### Predicted Class: `{label}`")

    # ---------- Show Activations ----------
    for name, tensor in activation_registry.items():
        st.subheader(f"Layer: {name}")
        tensor = tensor.squeeze(0)  # remove batch

        num_feature_maps = min(12, tensor.shape[0])
        fig, axs = plt.subplots(1, num_feature_maps, figsize=(15, 5))
        for i in range(num_feature_maps):
            feature_map = tensor[i].cpu().numpy()
            height, width = feature_map.shape
            axs[i].imshow(feature_map, cmap='viridis')
            axs[i].axis('off')
            axs[i].set_title(f"Map {i}")
            axs[i].set_title(f"Map {i}\n{height}×{width}")
        st.pyplot(fig)

    if not model_file:
        st.warning("Please upload a PyTorch model (.pth).")
    if uploaded_file and not model:
        st.warning("Model is not loaded.")


if __name__ == "__main__":
    main()
