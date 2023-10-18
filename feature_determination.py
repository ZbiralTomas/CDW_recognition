import os
import cv2
import tkinter as tk
from get_features import get_directory

class Features(object):
    def __init__(self, entropy, mig, r_rel, red, weight, volume):
        self.entropy = entropy
        self.mig = mig
        self.r_rel = r_rel
        self.red = red
        self.weight = weight







