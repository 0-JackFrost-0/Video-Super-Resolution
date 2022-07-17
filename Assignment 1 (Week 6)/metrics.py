# This file will store functions that will present a performance metric of whatever image processing is performed
#
#
# TASKS:-
# Implement metrics as required

import torch
from torch import nn
from math import log10
import numpy as np


def calc_psnr(img1, img2):
    mse = nn.MSELoss()
    return 10*log10(1/mse(img1, img2))
