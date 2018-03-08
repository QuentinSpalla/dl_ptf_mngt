#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 21:21:52 2018

@author: SPALLA
"""

import constants
from get_data import AllData

dj = AllData(constants.INPUT_FILE)
dj.add_indicators()

print('the end')