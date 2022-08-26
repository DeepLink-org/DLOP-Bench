from .types import SampleSource


class SampleConfig(object):
    """Standard execution config of a sample.

    Args:
        args_cases(list): Multiple sets of hyperparameters used to generate
            numpy data args.
        requires_grad(list[bool]): The requires_grad state of tensors needed in
            sample function input args.
        backward(bool | list(bool)): This indicates whether the sample function
            output do backward.
        warm_up_iters(int): The warm up running iters before record sample time
            performance.
        performance_iters(int): The running iters when record sample time 
            performance.
        timeline_iters(int): The running iters when generate timeline profile file.
        save_timeline(bool): This indicates whether to generate timeline profile
            file.
        rtol(float): The rtol arg used in torch or numpy allclose function.
        atol(float): The atol arg used in torch or numpy allclode function.
        source(SampleSource): Which framework samples come from.
        url(str): Where samples come from.
        tags(list[SampleTag]): Features the sample have.
    """
    def __init__(self,
                 args_cases=[],
                 requires_grad=[],
                 backward=False,
                 warm_up_iters=10,
                 performance_iters=1000,
                 timeline_iters=10,
                 save_timeline=False,
                 rtol=1e-04,
                 atol=1e-04,
                 source=SampleSource.UNKNOWN,
                 url=None,
                 tags=[]):
        assert len(args_cases) > 0
        self._args_cases = args_cases
        self._requires_grad = requires_grad
        self._backward = backward
        self._warm_up_iters = warm_up_iters
        self._performance_iters = performance_iters
        self._timeline_iters = timeline_iters
        self._save_timeline = save_timeline
        self._rtol = rtol
        self._atol = atol
        self._source = source
        self._url = url
        self._tags = tags

    @property
    def args_cases(self):
        return self._args_cases

    @property
    def requires_grad(self):
        return self._requires_grad

    @property
    def backward(self):
        return self._backward

    @property
    def warm_up_iters(self):
        return self._warm_up_iters

    @property
    def performance_iters(self):
        return self._performance_iters

    @property
    def timeline_iters(self):
        return self._timeline_iters

    @property
    def save_timeline(self):
        return self._save_timeline

    @property
    def rtol(self):
        return self._rtol

    @property
    def atol(self):
        return self._atol

    @property
    def source(self):
        return self._source

    @property
    def tags(self):
        return self._tags

    def show_info(self):
        tags = ""
        for tag in self._tags:
            tags = tags + tag.value + " "
        return self._source.value, self._url, tags

    def show(self):
        """Print sample source, url and tags.
        """
        source, url, tags = self.show_info()
        print("source:", source)
        print("url:", url)
        print("tags:", tags)
