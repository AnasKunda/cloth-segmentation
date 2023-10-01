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
import subprocess
#from ultralytics import YOLO
from PIL import Image, ImageOps
from predict import predict

LOGGER = get_logger(__name__)

def run():
    st.set_page_config(
        page_title="Cloth Segmentation",
        page_icon="👋",
    )

    subprocess.run([f"{sys.executable}", "preDeploy.sh"])

    st.write("# Cloth Segmentation Project")

    # st.sidebar.success("Select a demo above.")

    uploaded_file = st.file_uploader(label="Upload image")

    if uploaded_file:
        img = Image.open(uploaded_file)
        img = ImageOps.exif_transpose(img)
        st.image(img)
        predict(img)


if __name__ == "__main__":
    run()
