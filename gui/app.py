import streamlit as st
from streamlit_image_select import image_select
import streamlit_image_coordinates
from PIL import Image, ImageDraw

st.header("Retinal Image Segmentation")
st.write("Click on the image to select the region of interest and label it as 'Vein' or 'Artery'.")

def get_ellipse_coords(point: tuple[int, int]) -> tuple[int, int, int, int]:
    center = point
    radius = 10
    return (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )

img = image_select(label = "Select an image",
                   images = [r"IR_Case_011.png", r"IR_Case_017.png", r"IR_Case_019.png", r"IR_Case_020.png"],
                   captions = ["IR_Case_011", "IR_Case_017", "IR_Case_019", "IR_Case_020"],
                   key = "img_select")

col1, col2 = st.columns(2)

with col1:
    draw = ImageDraw.Draw(Image.open(img))

    # Draw an ellipse at each coordinate in points
    for point in st.session_state["points"]:
        coords = get_ellipse_coords(point)
        draw.ellipse(coords, fill="red")

    value = streamlit_image_coordinates(img, key="pil")

    if value is not None:
        point = value["x"], value["y"]

        if point not in st.session_state["points"]:
            st.session_state["points"].append(point)
            st.rerun()

    st.write("Points:", st.session_state["points"])

with col2:
    st.image(img)

# Define the path to your images directory
image_dir = 'train/training_images'

# List your image files
image_files = [
    'IR_Case_011.png',
    'IR_Case_017.png',
    'IR_Case_019.png',
    'IR_Case_020.png'
]

# Display images using Streamlit
for image_file in image_files:
    image_path = f"{image_dir}/{image_file}"
    try:
        image = Image.open(image_path)
        st.image(image, caption=image_file)
    except FileNotFoundError:
        st.error(f"Image file not found: {image_path}")
