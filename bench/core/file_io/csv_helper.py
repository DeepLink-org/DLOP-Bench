import os
import csv
from bench.common.settings import BENCH_DEBUG


class CsvHelper(object):
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
        content = csv.reader(self._file_path)
        return content

    def save(self, content):
        csv.DictWriter(content, fieldnames=content.keys())
