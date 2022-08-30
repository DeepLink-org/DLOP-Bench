# Copyright (c) OpenComputeLab. All Rights Reserved.

import os
from bench.core.file_io.json_helper import JsonHelper


class TestJsonHelper(object):
    def _gen_json_file_path(self):
        return os.path.join(os.getcwd(), "test.json")

    def _prepare_json_helper(self):
        return JsonHelper(self._gen_json_file_path())

    def test_creat(self):
        self._prepare_json_helper()
        assert os.path.exists(self._gen_json_file_path())
        os.remove(self._gen_json_file_path())

    def test_read(self):
        res = self._prepare_json_helper()
        res_content = res.read()
        assert res_content == {}
        os.remove(self._gen_json_file_path())

    def test_save(self):
        res = self._prepare_json_helper()
        dic = {"name": "json", "time": 1850}
        res.save(dic)
        result = res.read()
        assert result == {"name": "json", "time": 1850}
        assert result["time"] == 1850
        assert result["name"] == "json"
        os.remove(self._gen_json_file_path())
