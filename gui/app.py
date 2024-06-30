import streamlit as st
import pandas as pd
import numpy as np
import random
import torch
from streamlit_image_select import image_select
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw
from io import BytesIO
from util import model_predict_click, model_predict_box, model_predict_everything, show_click, show_everything, get_color

st.set_page_config(layout="wide",
               initial_sidebar_state="expanded",
               page_title="Retinal Image Segmentation - SegRAVIR and Segment Anything Model",
               page_icon="👁️")

st.header("👁️ Retinal Image Segmentation")
st.write("Click on the image to select the region of interest and label it as 'Vein' or 'Artery'.")

with st.sidebar:
    im = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])
    model = st.selectbox("Select a Segment Anything model", ["vit_b", "vit_l"])
    show_mask = st.checkbox("Show mask", value=True)
    radius_width = st.slider('Radius for points',0,20,5,1)

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

def click(container_width,height,scale,radius_width,show_mask,model,im):
    for each in ['color_change_point_box','input_masks_color_box']:
        if each in st.session_state:st.session_state.pop(each)
    canvas_result = st_canvas(
            fill_color="rgba(255, 255, 0, 0.8)",
            background_image = st.session_state['im'],
            drawing_mode='point',
            width = container_width,
            height = height * scale,
            point_display_radius = radius_width,
            stroke_width=2,
            update_streamlit=True,
            key="click",)
    if not show_mask:
        im = Image.fromarray(im).convert("RGB")
        rerun = False
        if im != st.session_state['im']:
            rerun = True
        st.session_state['im'] = im
        if rerun:
            st.experimental_rerun()
    elif canvas_result.json_data is not None:
        color_change_point = st.button('Save color')
        df = pd.json_normalize(canvas_result.json_data["objects"])
        if len(df) == 0:
            st.session_state.clear()
            if 'canvas_result' not in st.session_state:
                st.session_state['canvas_result'] = len(df)
                st.experimental_rerun()
            elif len(df) != st.session_state['canvas_result']:
                st.session_state['canvas_result'] = len(df)
                st.experimental_rerun()
            return
        
        df["center_x"] = df["left"]
        df["center_y"] = df["top"]
        input_points = []
        input_labels = []
        
        for _, row in df.iterrows():
            x, y = row["center_x"] + 5, row["center_y"]
            x = int(x/scale)
            y = int(y/scale)
            input_points.append([x, y])
            if row['fill'] == "rgba(0, 255, 0, 0.8)":
                input_labels.append(1)
            else:
                input_labels.append(0)
        
        if 'color_change_point' in st.session_state:
            p = st.session_state['color_change_point']
            if len(df) < p:
                p = len(df) - 1
                st.session_state['color_change_point'] = p
            masks = model_predict_click(im,input_points[p:],input_labels[p:],model)
        else:
            masks = model_predict_click(im,input_points,input_labels,model)
        
        if color_change_point:
            st.session_state['color_change_point'] = len(df)
            st.session_state['input_masks_color'].append([np.array([]),np.array([])])
        else:
            color = np.concatenate([random.choice(get_color()), np.array([0.6])], axis=0)
            if 'input_masks_color' not in st.session_state:
                st.session_state['input_masks_color'] = [[masks,color]]
            
            elif not np.array_equal(st.session_state['input_masks_color'][-1][0],masks):
                st.session_state['input_masks_color'][-1] = [masks,color]
            im_masked = show_click(st.session_state['input_masks_color'])
            im_masked = Image.fromarray(im_masked).convert('RGBA')
            im = Image.alpha_composite(Image.fromarray(im).convert('RGBA'),im_masked).convert("RGB")
            torch.cuda.empty_cache()
            rerun = False
            if im != st.session_state['im']:
                rerun = True
            st.session_state['im'] = im
            if rerun:
                st.experimental_rerun()
        im_bytes = BytesIO()
        st.session_state['im'].save(im_bytes,format='PNG')
        st.download_button('Download image',data=im_bytes.getvalue(),file_name='seg.png')

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

if im is not None:
    img = im

st.write(img.shape)
width, height   = img.shape[:2]
im              = np.array(im)
container_width = 700
scale           = container_width/width

click(container_width,height,scale,radius_width,show_mask,model,im)

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