import streamlit as st
from streamlit_image_select import image_select
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image, ImageDraw

st.set_page_config(layout="wide",
               initial_sidebar_state="collapsed",
               page_title="Retinal Image Segmentation - SegRAVIR and Segment Anything Model",
               page_icon="👁️")

st.header("Retinal Image Segmentation")
st.write("Click on the image to select the region of interest and label it as 'Vein' or 'Artery'.")

if "points" not in st.session_state:
    st.session_state["points"] = []

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
                   images = [r"gui/train/training_images/IR_Case_011.png", 
                             r"gui/train/training_images/IR_Case_017.png", 
                             r"gui/train/training_images/IR_Case_019.png", 
                             r"gui/train/training_images/IR_Case_020.png",
                             r"gui/train/training_images/IR_Case_021.png",
                             r"gui/train/training_images/IR_Case_022.png",],
                   captions = ["IR_Case_011", "IR_Case_017", "IR_Case_019", "IR_Case_020", "IR_Case_021", "IR_Case_022"],
                   key = "img_select")

# col1, col2 = st.columns(2)

# with col1:
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

# with col2:
#     st.image(img)