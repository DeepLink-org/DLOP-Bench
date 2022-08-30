# Copyright (c) OpenComputeLab. All Rights Reserved.

from .case_fetcher import CaseFetcher


class Registry(object):
    """The object to manager sample CaseFetcher.
    """
    def __init__(self):
        self._case_fetchers = dict()

    def register(self, name=None, content=None):
        """Register sample CaseFetcher to dict.
        """
        if name is not None and isinstance(content, CaseFetcher):
            assert (name not in self._case_fetchers
                    ), "Case {} already exists.".format(name)
            self._case_fetchers[name] = content
        else:

            def wrapper(name, content):
                self._case_fetchers[name] = content

            return wrapper

    def get(self, name):
        """Get CaseFetcher of one sample.
        
        Args:
            name(str): Sample name.
        Returns:
            CaseFetcher: CaseFetcher."""
        self.exists(name)
        return self._case_fetchers[name]

    def exists(self, name):
        """Whether the sample exists."""
        assert isinstance(name, str)
        assert name in self._case_fetchers, "Case {} not exists.".format(name)

    def key_iters(self):
        """Sample name iters."""
        return iter(sorted(self._case_fetchers))


registry = Registry()
