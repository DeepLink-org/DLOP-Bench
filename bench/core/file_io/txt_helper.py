import os
import txt
import shutil
from bench.common.settings import BENCH_DEBUG


class TxtHelper(object):
    def __init__(self, dir_path):
        self._dir_path = dir_path
        self.create()

    def create(self):
        if os.path.exists(self._dir_path):
            shutil.rmtree(self._dir_path)
        os.mkdir(self._dir_path)
        
        if not os.path.exists(self._dir_path):
            raise Exception("create {} failed.".format(self._dir_path))

    def read(self):
        content = None
        with open(self._file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content

    def save(self, case_name, content):
        with open(self._dir_path + "/" + case_name + ".txt", "w") as f:
            f.write(content)
