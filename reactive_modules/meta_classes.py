import abc

_instance_stor: dict[type, any] = {}


class Singleton(type):
    def __init__(cls, name, bases, dict):
        super(Singleton, cls).__init__(name, bases, dict)
        cls.instance = None

    def __call__(cls):
        if cls not in _instance_stor:
            _instance_stor[cls] = super(Singleton, cls).__call__()
        return _instance_stor[cls]


class SingletonFromAbc(Singleton, abc.ABCMeta):
    def __new__(mcls, name, bases, namespace, **kwargs):
        def __eq__(self, other):
            return self is other

        if __eq__.__name__ not in namespace:
            namespace[__eq__.__name__] = __eq__

        def __hash__(self):
            return id(self)

        if __hash__.__name__ not in namespace:
            namespace[__hash__.__name__] = __hash__

        return super().__new__(mcls, name, bases, namespace, **kwargs)
