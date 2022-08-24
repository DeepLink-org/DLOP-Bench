import os
import shutil
import torch
from bench.tools import grip


@grip(dist_dir="./")
def func_to_grip(a, b):
    return a + 1, b + 1


@grip(dist_dir="./", class_entry="forward")
class ClassToGrip(object):

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return a + 1, b + 1


class TestGripper(object):

    def test_grip_func(self):
        _, _ = func_to_grip(torch.zeros((2, 2)), 1)
        assert os.path.exists("./func_to_grip/__init__.py")
        assert os.path.exists("./func_to_grip/pat_impl.py")
        shutil.rmtree("./func_to_grip/")

    def test_grip_class(self):
        t = ClassToGrip()
        t.forward(torch.zeros((2, 2)), 1)
        assert os.path.exists("./forward/__init__.py")
        assert os.path.exists("./forward/pat_impl.py")
        shutil.rmtree("./forward/")
