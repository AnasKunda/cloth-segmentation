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
from ultralytics.engine.results import Results
import numpy as np
from PIL import Image as im
from io import BytesIO
from typing import Dict


def show_code(demo):
    """Showing the code of the demo."""
    show_code = st.sidebar.checkbox("Show code", True)
    if show_code:
        # Showing the code of the demo.
        st.markdown("## Code")
        sourcelines, _ = inspect.getsourcelines(demo)
        st.code(textwrap.dedent("".join(sourcelines[1:])))

def predict(img: im) -> Results:
    """
    predict segments using YOLO.

    Parameters
    ----------
    img: PIL image
        image object

    Returns
    -------
    Results
        a dict-type object containing segmentation results
    """
    model = YOLO('best.pt')
    res = model(img)
    st.image(res[0].plot())
    return res

def extract_segment(res: Results) -> Dict[str, BytesIO]:
    """
    extract segments using predicted result.

    Parameters
    ----------
    res: Results
        dict-type object containing segmentation results

    Returns
    -------
    Dict[str, BytesIO]
        A dictionary containing class labels as key, and image in BytesIO format as value
    """
    no_of_segments = 0
    images = {}
    for r in res[0]:
        no_of_segments += 1
        # get class name of predicted segment from results
        class_label = r.names[r.boxes.cls.item()] + '_' + str(no_of_segments)

        # fetch segmentation as a numpy array
        masks_data = r.numpy().masks.data[0]

        # resize masks to fit original image size
        resized_mask = scale_image(masks_data, r.orig_img.shape)

        # stack 3 layers of same numpy mask. Each layer will be multiplied with image's respective R, G or B layer.
        stacked = np.stack((resized_mask, resized_mask, resized_mask), axis=-2)
        stacked = stacked[:,:,:,0]

        # multiply segmented mask with image. pixels other than segmented part will become 0 due to multiplication.
        extracted = np.multiply(r.orig_img, stacked)

        # post-process image
        alpha = np.sum(extracted, axis=-1) > 0
        alpha = np.uint8(alpha * 255)
        final = np.dstack((extracted, alpha))

        # convert numpy to PIL image
        image = im.fromarray(final.astype('uint8'))
        buf = BytesIO()

        # save image in BytesIO format to make it downloadable using streamlit's download widget.
        image.save(buf, format='PNG')
        byte_im = buf.getvalue()
        
        # fill class label as key, and BytesIO image as value in dictionary object
        images[class_label] = byte_im
    return images   