# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pip install ultralytics
# sudo apt-get update && sudo apt-get install libgl1

import streamlit as st
from streamlit.logger import get_logger
import sys
from PIL import Image, ImageOps
from utils import predict, extract_segment

LOGGER = get_logger(__name__)

def run():
    st.set_page_config(
        page_title="Cloth Segmentation",
        page_icon=":dress:",
    )

    st.write("# Cloth Segmentation Project")
    # file uploaded widget
    uploaded_file = st.file_uploader(label="Upload image")

    if uploaded_file:
        img = Image.open(uploaded_file)
        # Removing any exif orientation TAG to avoid auto rotation of image
        img = ImageOps.exif_transpose(img)
        # Display the image
        st.image(img)
        # predict segments
        res = predict(img)
        # extract predicted segments
        segmented_images = extract_segment(res)
        # add a download button for each predicted segment
        for class_label, image in segmented_images.items():
            c = class_label.split('_')[0]
            st.download_button(
                label = f'Download {c} segment in PNG',
                data = image,
                file_name = f'segment_{c}.png',
                mime = 'image/png'
            )


if __name__ == "__main__":
    run()
