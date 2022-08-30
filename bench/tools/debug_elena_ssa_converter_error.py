# Copyright (c) OpenComputeLab. All Rights Reserved.

from elena.ssa_converter.interface import dump_config


with open("coderop_0.json", "r") as f:
    config = f.read()

print(config)
dump_config.dump_graph_config(config)
