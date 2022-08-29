# Copyright (c) OpenComputeLab. All Rights Reserved.
import os
from bench.common import auto_register

auto_register(os.path.dirname(os.path.realpath(__file__)), __name__)
