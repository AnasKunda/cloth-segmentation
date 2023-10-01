import streamlit as st
from ultralytics import YOLO

def predict(img):
    model = YOLO('best.pt')
    res = model(img)
    st.image(res[0].plot())
    return