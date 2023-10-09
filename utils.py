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

import inspect
import textwrap

import streamlit as st

import streamlit as st
from ultralytics import YOLO

from ultralytics.utils.ops import scale_image
import numpy as np
from PIL import Image as im
from io import BytesIO


def show_code(demo):
    """Showing the code of the demo."""
    show_code = st.sidebar.checkbox("Show code", True)
    if show_code:
        # Showing the code of the demo.
        st.markdown("## Code")
        sourcelines, _ = inspect.getsourcelines(demo)
        st.code(textwrap.dedent("".join(sourcelines[1:])))

def predict(img):
    model = YOLO('best.pt')
    res = model(img)
    st.image(res[0].plot())
    return res

def extract_segment(res):
    
    no_of_segments = 0
    images = {}
    for r in res[0]:
        no_of_segments += 1
        class_label = r.names[r.boxes.cls.item()] + '_' + str(no_of_segments)
        masks_data = r.numpy().masks.data[0]
        resized_mask = scale_image(masks_data, r.orig_img.shape)

        stacked = np.stack((resized_mask, resized_mask, resized_mask), axis=-2)
        stacked = stacked[:,:,:,0]

        extracted = np.multiply(r.orig_img, stacked)

        alpha = np.sum(extracted, axis=-1) > 0
        alpha = np.uint8(alpha * 255)
        final = np.dstack((extracted, alpha))

        image = im.fromarray(final.astype('uint8'))
        buf = BytesIO()
        image.save(buf, format='PNG')
        byte_im = buf.getvalue()
        images[class_label] = byte_im
    return images   