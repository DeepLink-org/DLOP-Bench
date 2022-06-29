import os
import json
from long_tail_bench.common.settings import BENCH_DEBUG


class JsonHelper(object):
    def __init__(self, file_path):
        self._file_path = file_path
        self.create()

    def create(self):
        dir_path = os.path.dirname(self._file_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        if os.path.exists(self._file_path):
            os.remove(self._file_path)

        with open(self._file_path, "w", encoding="utf-8") as f:
            f.write("{}")
        if not os.path.exists(self._file_path):
            raise Exception("create {} failed.".format(self._file_path))

    def read(self):
        content = None
        with open(self._file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        return content

    def save(self, content):
        with open(self._file_path, "w") as f:
            json.dump(content, f)
        if BENCH_DEBUG:
            print("{} saved successfully!".format(self._file_path))
