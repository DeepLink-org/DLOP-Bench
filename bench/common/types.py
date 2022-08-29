from enum import Enum


class FrameType(Enum):
    Parrots = "parrots"
    Torch = "torch"
    XLA = "xla"
    JAX = "jax"


class PatExecMode(Enum):
    SYNC = "sync"
    ASYNC = "async"


class PatModes(Enum):
    S1 = "PatS1"  # parrots eager mode
    S2 = "PatS2"  # parrots jit mode of fixed shape
    S3 = "PatS3"  # parrots elena mode of fixed shape
    S4 = "PatS4"  # parrots jit mode of changed shape
    S5 = "PatS5"  # parrots elena mode of changed shape


class TorchModes(Enum):
    S1 = "ScriptS1"  # torch eager mode
    S3 = "ScriptS3"  # torch script mode


class TFModes(Enum):
    S1 = "XLAS1"  # tensorflow eager mode
    S3 = "XLAS3"  # tensorflow xla mode
    
class JAXModes(Enum):
    S1 = "JAXS1"  # tensorflow eager mode
    S3 = "JAXS3"  # tensorflow xla mode

class SampleSource(Enum):
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
    PARROTS_EXAMPLE = "parrots.example"
    MMEDIT = "mmediting"
    MMCLS = "mmclassification"
    FAIRSEQ = "fairseq"
    OPENNMT = "opennmt"
    ASSEMBLYAI = "assemblyai"
    PADDLEREC = "paddlerec"


class SampleTag(Enum):
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
