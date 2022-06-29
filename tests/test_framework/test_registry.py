from tests.test_framework.one_sample import test_sample_name, registry


class TestRegirtry(object):
    def test_get(self):
        res_case_fetcher = registry.get(test_sample_name)
        res_g_c_conv = res_case_fetcher.sample_config_getter()

        assert res_g_c_conv.args_cases == [(1, 1), (2, 3), (3, 3)]
        assert res_g_c_conv.requires_grad == [True, False]
        assert res_g_c_conv.backward == [True]
        assert res_g_c_conv.performance_iters == 1000
        assert res_g_c_conv.save_timeline

    def test_key_iters(self):
        count = 0
        for _ in registry.key_iters():
            count = count + 1
        assert count > 0
