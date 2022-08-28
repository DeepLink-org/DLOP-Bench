import os
import csv
from bench.common.settings import BENCH_DEBUG


class CsvHelper(object):
    def __init__(self, dir_path):
        self._dir_path = dir_path
        self.create()

    def create(self):
        if os.path.exists(self._dir_path):
            os.rmdir(self._dir_path)
        os.mkdir(self._dir_path)
        
        if not os.path.exists(self._dir_path):
            raise Exception("create {} failed.".format(self._dir_path))

    def read(self, case_name):
        content = None
        with open(self._dir_path + "/" + case_name + ".csv", "r", encoding="utf-8") as f:
            content = csv.DictReader(self._file_path)
        return content

    def save(self, case_name, content):
        with open(self._dir_path + "/" + case_name + ".csv", "w") as f:
            writer=csv.writer(f)
            for key, value in content.items():
                writer.writerow([key, value])
