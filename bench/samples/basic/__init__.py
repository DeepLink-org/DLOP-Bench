import os
from long_tail_bench.common import auto_register

auto_register(os.path.dirname(os.path.realpath(__file__)), __name__)
