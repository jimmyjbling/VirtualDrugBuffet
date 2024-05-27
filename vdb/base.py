import inspect
from copy import deepcopy

from vdb import __version__


def compile_step(cls):
    """
    This is a class decorator used on any class that we expect to pass into a Pipeline.
    Things like `BaseFPFunc` would not need this decorator as they are never instantiated as objects.
    Things like `ECFP4` would need this decorator, since it is intended to get used in Pipeline.
    """
    original_init = cls.__init__
    init_signature = inspect.signature(original_init)

    def new_init(self, *args, **kwargs):
        parameters = init_signature.parameters
        init_args = {param: arg for param, arg in zip(parameters, args)}
        init_args.update(kwargs)
        self._step_args = init_args
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls


# TODO this doesn't work, see issue #1
# def make_into_step(obj: object) -> object:
#     if not isinstance(obj, BaseEstimator):
#         raise TypeError("potential Steps must be an instance of `sklearn.base.BaseEstimator`")
#     if isinstance(obj, Step):
#         return obj
#     _mro = list(obj.__class__.__mro__)[:-1] + [Step, object]
#     _attribute_dict = {key: val for key, val in inspect.getmembers(obj.__class__)}
#     _new_class = type(obj.__class__.__name__+"Step", tuple(_mro), _attribute_dict)
#     _new_class = compile_step(_new_class)
#     _new_obj = _new_class()
#     for key, val in obj.__dict__.items():
#         setattr(_new_obj, key, val)
#     return _new_obj


class Step:
    """
    Steps are the most basic object in VDB.
    Anything intended to operate on chemicals in a Pipeline should be an instance of Step

    This doesn't mean everything is a Step.
    For example, `CurationSteps` are not Steps, since they are not intended for use on chemicals individually.
    However, `CurationWorkflow` (which is a collection of `CurationSteps`) is a Step,
    since it is intended to be used in a Pipeline on chemicals

    There are two goals with the Step object:
    1. Save and easily record the exact settings used to define a Step in the Pipeline
    2. Make Steps comparable to each other (to see if they do the same thing)

    The reasoning behind 2 becomes clear with the introduction of the "DataFlow" class.
    In order for the DataFlow to work effectively, it needs to know if Steps are the same; the Step class provides that
     capability
    """
    _step_args = {}

    def to_param_dict(self):
        return {"class": self.__class__.__name__, "vdb_version": __version__, "kwargs": self._step_args}

    def copy(self):
        return deepcopy(self)

    def __copy__(self):
        return self.copy()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.to_param_dict() == other.to_param_dict()
