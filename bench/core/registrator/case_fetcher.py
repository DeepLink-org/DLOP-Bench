class CaseFetcher(object):
    def __init__(self,
                 executer_creator,
                 sample_config_getter,
                 np_args_generator=None):
        self._executer_creator = executer_creator
        self._sample_config_getter = sample_config_getter
        self._np_args_generator = np_args_generator

    @property
    def executer_creator(self):
        return self._executer_creator

    @property
    def sample_config_getter(self):
        return self._sample_config_getter

    @property
    def np_args_generator(self):
        return self._np_args_generator
