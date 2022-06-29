import inspect


class CustomClassManager(object):
    _instance = None
    _custom_classes = {}

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance

    @classmethod
    def register_class(cls, cls_to_register):
        assert inspect.isclass(cls_to_register)
        if cls_to_register not in cls._custom_classes.values():
            cls._custom_classes[cls_to_register.__name__] = cls_to_register

    @classmethod
    def name_to_custom_class(cls):
        return cls._custom_classes

    @classmethod
    def get_custom_classes(cls):
        return cls._custom_classes.values()


custom_class_manager = CustomClassManager()
