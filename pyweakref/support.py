from ._internals import *
from ._internals import __all__ as _internals_all

import _collections_abc
import typing

__all__ = _internals_all + ["_remove_dead_pyweakref", "proxy", "register"]
__doc__ = "Python weak reference support"

# Bulk of the module

def _remove_dead_pyweakref(dct: _collections_abc.MutableMapping, key: typing.Any) -> None:
    """ Atomically remove key from dict if it points to a dead pyweakref. """
    value = dct[key]
    if not isinstance(value, ref):
        return
    if value() is None:
        del dct[key]
        
def proxy(obj: typing.Any, callback: _collections_abc.Callable = None) -> typing.Union[ProxyType, CallableProxyType]:
    "Return a weak proxy to obj."
    if callable(obj):
        return CallableProxyType(obj, callback)
    return ProxyType(obj, callback)

def register(cls: type) -> type:
    """Decorator to enable pyweakref.ref on a class."""
    try:
        if cls is ref:
            raise TypeError
        cls.__pyweakref__ = ReferenceDescriptor()
        return cls
    except Exception as e:
        message = "Cannot enable pyweakref.ref for class {0.__module__}.{0.__qualname__}".format(cls)
        exc = TypeError(message)
        exc = exc.with_traceback(e.__traceback__)
        raise exc    