# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 20:13:24 2024

@author: karthik.venkat
"""
import torch

status = torch.cuda.is_available()
print(status)