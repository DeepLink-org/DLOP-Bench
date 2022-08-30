# Copyright (c) OpenComputeLab. All Rights Reserved.

from enum import Enum


class FrameType(Enum):
    """Framework backend type.
    """
    Torch = "torch"
    XLA = "xla"
    JAX = "jax"

class TorchModes(Enum):
    """Pytorch backend execution stages.
    """
    S1 = "ScriptS1"  # torch eager mode
    S3 = "ScriptS3"  # torch script mode


class TFModes(Enum):
    """Tensorflow backend execution stages.
    """
    S1 = "XLAS1"  # tensorflow eager mode
    S3 = "XLAS3"  # tensorflow xla mode
    
class JAXModes(Enum):
    """JAX backend execution stages.
    """
    S1 = "JAXS1"  # tensorflow eager mode
    S3 = "JAXS3"  # tensorflow xla mode

class SampleSource(Enum):
    """The framework where samples come from.
    """
    UNKNOWN = "unknown"
    MMCV = "mmcv"
    MMDET = "mmdetection"
    MMSEG = "mmsegmentation"
    MMPOSE = "mmpose"
    MMACTION2 = "mmaction2"
    MMTRACKING = "mmtracking"
    SEGMENTBASE2 = "segmentbase2"
    POD = "pytorch-object-detection"
    GYM = "gym"
    SINGLE_REPO = "single-repo"
    MMEDIT = "mmediting"
    MMCLS = "mmclassification"
    FAIRSEQ = "fairseq"
    OPENNMT = "opennmt"
    ASSEMBLYAI = "assemblyai"
    PADDLEREC = "paddlerec"


class SampleTag(Enum):
    """Sample features.
    """
    InputAware = "InputAware"
    ViewAttribute = "ViewAttribute"
    IfElseBranch = "IfElseBranch"
    ForLoop = "ForLoop"
    AdvancedIndexing = "AdvancedIndexing"
    BuiltInDataStructure = "BuiltInDataStructure"
    Reduce = "Reduce"
    Customized = "Customized"
    Broadcast = "Broadcast"
    ThirdPartyCodes = "ThirdPartyCodes"
