# Cloth Segmentation using YOLOv8 and streamlit

This repo contains code of a streamlit web app used for cloth segmentation.

![image](https://github.com/AnasKunda/cloth-segmentation/assets/41727762/5445239c-e713-4f21-94b6-f10264f88166)

***

![cloth-segmentation-demo](https://github.com/AnasKunda/cloth-segmentation/assets/41727762/cc1b0408-80c4-49d2-af9e-970c63cb0982)

# Web App

The web app is created using Streamlit. (App URL: https://cloth-segmentation-216oli12ryc.streamlit.app/)

|  File  |  Description  |
|  ---   |  ---          |
|Hello.py| Main file running the web app. |
|utils.py| Utility functions used for data processing and inference. |
|best.pt | Trained Model Weights. |
|packages.txt| Linux packages required to be installed by streamlit's cloud. |
|requirements.txt | Python packages required to be installed. |

# Dataset

Fashionpedia's images and annotations are used. Annotations with latest file names are retrieved from [Kaggle version](https://www.kaggle.com/c/imaterialist-fashion-2020-fgvc7/data?select=train.csv).

Kaggle's annotations are in [RLE format](https://en.wikipedia.org/wiki/Run-length_encoding). These annotations are converted into YOLO format.

# Model

YOLOv8's segmentation model is used to train the data. The end-to-end process from data transformation to testing has been done seperately in Google Colab.
