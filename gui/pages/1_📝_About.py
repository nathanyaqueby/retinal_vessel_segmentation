import streamlit as st

st.title('📝 About the project')

st.write("""
        Done within the scope of a practical work assignment in the Institute for Machine Learning at [Johannes Kepler University Linz](www.jku.at).
        \

        👉 **Student:** [Nathanya Queby Satriani](https://www.linkedin.com/in/queby)
         \\
        👉 **Supervisor:** Hrvoje Bogunovic
        \\

        This project is a web application that allows users to segment retinal images using the Segment Anything model.
         """)

st.divider()

st.subheader('Abstract')

st.write("""
         Accurate segmentation of retinal vessels is crucial for diagnosing and monitoring various ophthalmic and systemic diseases. This report explores the application of human-in-the-loop segmentation techniques on the RAVIR dataset using SegRAVIR and the Segment Anything Model (SAM) developed by Meta AI. By incorporating human interactions where users can annotate arteries and veins through a Streamlit-based user interface, this study aims to enhance segmentation accuracy and reliability. Comparative analysis between the user-enhanced SAM and the original SegRAVIR model demonstrates the potential benefits of integrating human feedback into automated segmentation pipelines.
         """)

st.image("gui/pipeline.png", caption="Human-in-the-loop segmentation pipeline", use_column_width=True)