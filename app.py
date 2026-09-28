# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 16:42:32 2026

@author: User
"""

import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

st.set_page_config(page_title="Elephant Classifier", layout="centered")

st.title("Elephant Classifier")
st.write("Upload an image - the model will predict **African** or **Asian** elephants")