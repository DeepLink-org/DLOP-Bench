from .case_fetcher import CaseFetcher


class Registry(object):
    def __init__(self):
        self._case_fetchers = dict()

    def register(self, name=None, content=None):
        if name is not None and isinstance(content, CaseFetcher):
            assert (name not in self._case_fetchers
                    ), "Case {} already exists.".format(name)
            self._case_fetchers[name] = content
        else:

            def wrapper(name, content):
                self._case_fetchers[name] = content

            return wrapper

    def get(self, name):
        self.exists(name)
        return self._case_fetchers[name]

    def exists(self, name):
        assert isinstance(name, str)
        assert name in self._case_fetchers, "Case {} not exists.".format(name)

    def key_iters(self):
        return iter(sorted(self._case_fetchers))


registry = Registry()
