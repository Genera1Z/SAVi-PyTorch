import importlib
import pathlib as pl
import sys


MODULE_DICT = {}


class Config(dict):
    """Copy from easydict.whl. Support nested dict."""

    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)
        for k in self.__class__.__dict__.keys():
            flag1 = k.startswith("__") and k.endswith("__")
            flag2 = k in ("fromfile", "update", "pop")
            if any([flag1, flag2]):
                continue
            setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
        super(Config, self).__setattr__(name, value)
        super(Config, self).__setitem__(name, value)

    __setitem__ = __setattr__

    @staticmethod
    def fromfile(file_path: str) -> "Config":
        assert file_path.endswith(".py")
        file = pl.Path(file_path)
        assert file.is_file()
        file_dir = str(file.absolute().parent)
        fn = str(file.name).split(".")[0]
        sys.path.append(file_dir)
        module = importlib.import_module(fn)
        # cfg_dict = {
        #     k: v
        #     for k, v in module.__dict__.items()
        #     if not (k.startswith("__") and k.endswith("__"))
        # }
        cfg_dict = module.__dict__
        return Config(cfg_dict)

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(Config, self).pop(k, d)


def register_module(module, force=False):
    if not callable(module):
        raise TypeError(f"module must be Callable, but got {type(module)}")
    name = module.__name__
    if not force and name in MODULE_DICT:
        raise KeyError(f"{name} is already registered")
    MODULE_DICT[name] = module


def build_from_config(cfg: Config):
    """Build a module from config dict.

    Support two kinds of nesting:
    - {type:..,k:[{type:..,..},..],..}
    - {type:..,k:{type:..},..}
    """
    cfg = cfg.copy()
    cls_key = cfg.pop("type")
    # update cfg nestedly
    for key, value in cfg.items():
        # list of dict with type -- iterate and recur
        if (
            isinstance(value, (list, tuple))
            and len(value) > 0
            and isinstance(value[0], (Config, dict))
        ):
            assert all(isinstance(_, (Config, dict)) and "type" in _ for _ in value)
            value = [build_from_config(_) for _ in value]
        # dict with type -- recur
        elif isinstance(value, (Config, dict)) and "type" in value:
            value = build_from_config(value)
        cfg[key] = value
    # build obj with cfg
    obj = MODULE_DICT[cls_key](**cfg)
    return obj
