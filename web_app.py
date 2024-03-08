import streamlit as st
from streamlit_lottie import st_lottie


import torch
import torch.nn as nn

import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple
import requests
import os
from io import BytesIO


import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from io import BytesIO

# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="E-Waste Image Classification Project", page_icon=":recycle:",layout="wide")


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# local_css("C:/Users/shrikar/Downloads/final_major/style.css")

# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")


# ---- HEADER SECTION ----
with st.container():
    # Header
    st.title("E-Waste Image Classification Project")
    st.markdown("### Empowering Sustainability Through Deep Learning")
    st.subheader("Team: Rithvija, Shrikar, Valli :wave:")
    # Project description
    st.write(
        """In our E-Waste Image Classification Project, we leverage the power of transfer learning with Vision Transformer models to contribute towards a sustainable future. Our team, consisting of Rithvija, Shrikar, and Valli, collaborates to develop a robust image classification model for identifying electronic waste.

By utilizing transfer learning techniques with Vision Transformer models, we aim to expedite the process of training our classification model while still achieving high accuracy. Transfer learning allows us to leverage pre-trained Vision Transformer models that have been trained on large-scale image datasets, such as ImageNet, and fine-tune them on our specific electronic waste classification task. This approach not only accelerates the training process but also helps improve the generalization performance of our model.

Join us on this journey as we harness the capabilities of transfer learning and Vision Transformer models to address the environmental impact of electronic waste and pave the way towards a more sustainable future.""")

    st.write("[Learn More >](https://www.who.int/news-room/fact-sheets/detail/electronic-waste-(e-waste))")






# Define your device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your model and class names
model_save_path = "C:/Users/shrikar/Downloads/final_major/models/vision_transformer_image_classifier.pth"

class_names = ['keyboard','mobile','mouse','tv']
num_classes = len(class_names)

def create_vit_model(num_classes: int = 4, seed: int = 42):
    # Create ViT_B_16 pretrained weights, transforms and model
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.vit_b_16(weights=weights)

    # Freeze all layers in model
    for param in model.parameters():
        param.requires_grad = False

    # Change classifier head to suit our needs (this will be trainable)
    torch.manual_seed(seed)
    model.heads = nn.Sequential(nn.Linear(in_features=768, out_features=num_classes))  # update to reflect target number of classes

    return model, transforms


model, _ = create_vit_model(num_classes=num_classes)

# Load the saved model state dictionary
state_dict = torch.load(model_save_path)

# Create a new dictionary and copy the state dictionary's keys and values to the new dictionary
modified_state_dict = {}
for key, value in state_dict.items():
    modified_key = key.replace("heads.weight", "heads.0.weight").replace("heads.bias", "heads.0.bias")
    modified_state_dict[modified_key] = value

# Load the modified state dictionary into the model
model.load_state_dict(modified_state_dict, strict=False)


# Ensure the model is in evaluation mode
model.eval()

print("Model loaded successfully.")


def pred_and_plot_image(
    model: torch.nn.Module,
    class_names: List[str],
    img: Image.Image,  # Change image_path to img of type Image.Image
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
    device: torch.device = device,
):
    # Remove the part that opens the image from a file

    # Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    ### Predict on image ###

    # Make sure the model is on the target device
    model.to(device)

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.no_grad():
        # Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image.to(device))

    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Check if predicted probability is greater than 0.5
    if target_image_pred_probs.max() > 0.8:
        # Plot image with predicted label and probability
        plt.figure()
        plt.imshow(img)
        plt.title(
            f"Predicted Class: {class_names[target_image_pred_label]} | Probability(%): {target_image_pred_probs.max()*100:.2f}%"
        )
        plt.axis(False)
        st.pyplot(plt)
    else:
        # Plot image with predicted label as 'None' since probability is less than 0.5
        plt.figure()
        plt.imshow(img)
        plt.title("Pred: None")
        plt.axis(False)
        st.pyplot(plt)

def main():
    st.title("E-Waste Image Prediction")
#     st.sidebar.title("Options")
    st.title("Options")
    option = st.radio("Select Input Method", ["Upload File", "Enter URL"])

    if option == "Upload File":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            
    elif option == "Enter URL":
        url = st.text_input("Enter Image URL:")
        if url:
            img = download_image_from_url(url)
            if img is not None:
                st.image(img, caption='Image from URL.', use_column_width=True)
                st.write("")
                

    if st.button("Predict"):
        # st.write("Classifying...")
        pred_and_plot_image(model, class_names, img)
        

def download_image_from_url(url: str) -> Image.Image:
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        return img
    else:
        st.error("Failed to load image from URL.")
        return None


if __name__ == "__main__":
    main()

    
    
    
with st.container():
    st.write("---")
    st.header("Classes in Dataset")
    
    # Define image URLs for each class
    keyboard_image_url = "https://thumbs.dreamstime.com/b/burning-keyboard-3318127.jpg"
    mobile_image_url = "https://media.umbraco.io/heartcore/hbzjy5sl/brokenphone.png?format=png"
    mouse_image_url = "https://c8.alamy.com/comp/CW0NKG/burnt-computer-mouse-CW0NKG.jpg"
    tv_image_url = "https://img.freepik.com/premium-photo/old-tv-set-exploding-with-fire-smoke_777271-4048.jpg"

    # Add Streamlit components without the CSS styles
    keyboard, mobile, mouse, tv = st.columns(4)

    with keyboard:
        st.header("Keyboard")
        st.image(keyboard_image_url, use_column_width=True)
        
    with mobile:
        st.header("Mobile")
        st.image(mobile_image_url, use_column_width=True)

    with mouse:
        st.header("Mouse")
        st.image(mouse_image_url, use_column_width=True)

    with tv:
        st.header("TV")
        st.image(tv_image_url, use_column_width=True)

