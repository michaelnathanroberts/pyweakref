# Internals. Do not import directly.

import _collections_abc
import gc
import sys
import threading
import types
import typing

try:
    import numpy.core.numerictypes as nptypes
    import numpy.core._multiarray_umath as np
except ImportError:
    np = nptypes = None
    
__all__ = [
    # Classes
    "AbstractProxyType",
    "CallableProxyType", 
    "ProxyType", 
    "ReferenceDescriptor", 
    "ReferenceType",
    
    # Functions
    "circular_reference_count",
    "disable_purging",
    "enable_purging",
    "get_pyweakref_count",
    "get_pyweakrefs",
    "purge",
    "purging",
    
    # Aliases
    "ref"
]

### Internal objects###

def _callback(ref):
    return _reference_registry[id(ref)][1]

# Marker for _get_circular_ref_count
_circular_ref_marker = object()

# Whitelist for _get_circular_ref_count
_circular_ref_whitelist = (
    # builtins
    bool, bytearray, bytes, complex, float, int, memoryview,
    object, range, slice, str, type(None), type(NotImplemented), type(Ellipsis), 
    # types
    types.BuiltinFunctionType, types.BuiltinMethodType,
    types.GetSetDescriptorType, types.MemberDescriptorType, types.MethodWrapperType, 
    types.WrapperDescriptorType,types.ClassMethodDescriptorType, types.CodeType, 
    types.FrameType, types.TracebackType,
)

def _descriptors(obj):
    # Get attributes of the data
    get_descriptor_dict = {}
    # object has no data descriptors, don't investigate it.
    reverse_mro = reversed(type(obj).__mro__)
    for cls in reverse_mro:
        if cls is object:
            continue
        for name, value in cls.__dict__.items():
            if _is_get_descriptor(value):
                get_descriptor_dict[name] = value
    return get_descriptor_dict

def _get_circular_ref_count(obj, data=_circular_ref_marker, memo=None, parent=None):
    
    ## Pre-examination checks, base cases ##
    # When invoked with only 1 argument (in _get_threshold),
    # a marker prevents an automatic return of 1. Now the
    # marker's purpose if fulfilled, set data to obj.
    if data is _circular_ref_marker:
        data = obj
    

    # Some instances are built-in types are known not 
    # to contain any references of obj. If so, save
    # precious time and return 0.
    if type(data) in _circular_ref_whitelist:
        return 0

    # Memo is set to None when invoked with 1 argument.
    # If so, set memo to an empty list (which can't be
    # corrupted).
    if memo is None:
        memo = []

    # Sometimes, we have already examined an object. All
    # of its references to obj have been exposed. Return 0,
    # there are no more left to expose.
    if id(data) in memo:
        return 0

    ## Preperation for examination of the data ##

    # Add data to memo. This prevents us from examining 
    # the data more than once.
    memo.append(id(data))
    
    # The counter of self-references
    counter = 0
    
    ## Investigate data's attributes through get descriptors and its iterator
    get_descriptor_dict = _descriptors(data)
    id2obj = {}
    referenced_objects = {}
    
    ## Collect attributes
    for name, get_descriptor in get_descriptor_dict.items():
        try:
            value = get_descriptor.__get__(data, type(data))
            id2obj[id(value)] = value
            referenced_objects.setdefault(id(value), 0)
            referenced_objects[id(value)] += 1
        except AttributeError:
            pass
        
    ### Collect iterator yields
    if isinstance(data, _collections_abc.Mapping):
        for key, value in data.items():
            # Key
            id2obj[id(key)] = key
            referenced_objects.setdefault(id(key), 0)
            referenced_objects[id(key)] += 1
            # Value
            id2obj[id(value)] = value
            referenced_objects.setdefault(id(value), 0)
            referenced_objects[id(value)] += 1
    elif isinstance(data, _collections_abc.Iterable):
        iterator = iter(data)
        try:
            while True:
                item = next(iterator)
                id2obj[id(item)] = item
                referenced_objects.setdefault(id(item), 0)
                referenced_objects[id(item)] += 1
        except Exception:
            pass          
    
    ### Investigate objects solely dependent on the data
    for id_, count in referenced_objects.items():
        value = id2obj[id_]
        # Add 1 if value is obj. Value must have
        # come from obj, because we have searching
        # obj and its dependents
        if value is obj:
            counter += count        
        # Recurse if value totally dependent on data
        if sys.getrefcount(value) - 3 <= count: # Subtract id2obj, value, param
            counter += _get_circular_ref_count(obj, value, memo, data)
     # We're done! Return the counter.
    return counter

def _is_eligible(obj):
    cls = type(obj)
    if cls is type:
        return True
    if isinstance(cls.__dict__.get("__pyweakref__", None), ReferenceDescriptor):
        return True
    return isinstance(cls.__dict__.get('__weakref__', None), types.GetSetDescriptorType)

def _is_get_descriptor(obj):
    if callable(obj):
        # obj is some function, not a get descriptor
        return False
    if isinstance(obj, (classmethod, staticmethod)):
        # obj is callable when used
        return False
    cls = type(obj)
    getfunc = getattr(cls, "__get__", None)
    return callable(getfunc)

def _numpy_circular_ref_count(obj):
    if isinstance(obj,  (nptypes.bool_, nptypes.number, nptypes.flexible)):
        return 0
    if isinstance(obj, np.ndarray):
        if issubclass(obj.dtype.type, (nptypes.bool_, nptypes.number, nptypes.flexible)):
            return 0
        print(list(obj))
        return _get_circular_ref_count(obj, list(obj), None, obj)
    return NotImplemented

# Whether purging is enabled. Starts off as False.
# Enabled in final touches.
_purge = False

def _purge_func(chain=True):

    # Number of weak references purged
    purged = 0

    # For every (id, ref) pair in the registry,
    # check if the number of references of ref()
    # is less or equal to the threshold for purging.
    # If so, purge the reference and add 1 to purged.
    #
    # To purge an reference means to delete it from
    # the registry and make it reference None instead
    # of its object. 
    # 
    # The object is then garbage collected (see below).
    for id_, ref_list in tuple(_reference_id_registry.items()):
        if not ref_list:
            continue
        ref = ref_list[0]
        obj = ref()
        count = sys.getrefcount(obj) - 2
        threshold = circular_reference_count(obj) + get_pyweakref_count(obj)
        if count <= threshold:
            for ref in ref_list:
                if callable(ref.__callback__):
                    ref.__callback__.__call__(ref)
                _reference_registry[id(ref)] = None, None
            purged += 1
            del _reference_id_registry[id(obj)] 

    # If a reference has been purged, run the garbage
    # collector now.
    if purged:
        gc.collect()

    # If purging has been enabled and the chain parameter is
    # True, then we schedule a call of this function in 5
    # seconds.
    if purging() and chain:
        global _purge_timer
        _purge_timer = threading.Timer(5.0, _purge_func)
        _purge_timer.start()        

# The purge timer. Starts at None,
# initialized and updated when purging is enabled.
_purge_timer = None 

# The type of ReferenceDescriptor's __doc__ attribute. ReferenceDescriptor's
# __doc__ displays one message without an instance and another with an instance.
# This is impossible without another descriptor (property uses 
# types.MemberDescriptorType for example, but we can't instantiate it).

class _reference_descriptor_doc_type(object):

    def __get__(self, instance, owner=None):
        "Return an attribute of instance, which is of type owner."
        if instance is None:
            # Result of ReferenceDescriptor.__doc__ 
            return """
            Descriptor which allows a pyweakref.ref to instances of a class,
            without using __weakref__.

            Unless a class has a __weakref__ descriptor, it should be register()ed. 
            It is bad programming practice to'register' a class without using register()."""
        else:
            # Result of r.__doc__ where isinstance(r, ReferenceDescriptor)
            return "list to the pure python weak references of the object"

    __slots__ = ()
    
def _referent(ref):
    return _reference_registry[id(ref)][0]

def _proxied(proxy):
    return _referent(_proxy_registry[id(proxy)])

def _proxy_callback(proxy):
    return _callback(_proxy_registry[id(proxy)])

## Registries ##

# proxy -> ref
_proxy_registry = {}

# id(referent) -> ref
_reference_id_registry = {}

# ref -> (referent, callback)
_reference_registry = {}


### Start of public API ###

# Rename for asthetic purposes.
__name__ = "support"

class AbstractProxyType(object):
    """
    Abstract base class for ProxyType and CallableProxyType. 
    
    Use 'isinstance(obj, AbstractProxyType)' to test if the 
    object is a proxy.
    """
    @property
    def __callback__(self):
        return _proxy_callback(self)

    @property
    def __class__(self):
        return _proxied(self).__class__

    def __getattribute__(self, name):
        try:
            referent = _proxied(self)
            return type(referent).__getattribute__(referent, name)
        except AttributeError:
            return object.__getattribute__(self, name)

    def __hash__(self):
        raise TypeError("unhashable object")

    def __new__(cls, obj, callback):
        if cls not in (CallableProxyType, ProxyType):
            message = """No subclass of AbstractProxyType,
            including AbstractProxyType itself, except 
            ProxyType (normal proxy) or CallableProxyType, 
            (callable proxy), may be initiated. """
            raise TypeError(message)
        ref = ReferenceType(obj, callback)
        self = object.__new__(cls)
        _proxy_registry[id(self)] = ref
        return self
    
    def __reduce__(self):
        raise TypeError("cannot pickle proxy object")
    
    def __reduce_ex__(self, protocol: int):
        raise TypeError("cannot pickle proxy object")    

    def __repr__(self):
        obj = _proxied(self)
        return f"<{type(self).__module__}.{type(self).__qualname__} object at 0x{hex(id(self))[2:].upper()}" \
               + f", to <{type(obj).__module__}.{type(obj).__qualname__} object " \
               + f"at 0x{hex(id(_proxied(obj)))[2:].upper()}>>"    

    __slots__ = ()      
    

class CallableProxyType(AbstractProxyType):
    """CallableProxyType(obj) -> callable weak proxy to obj.
    
    A callable weak proxy is not hashable no matter what, 
    because of its mutable nature.
    
    Besides these differences it is equivalent to obj,
    including the __class__ attribute. Passing a callable
    weak proxy into type() will still yield the proxy's
    type, not obj's type.
    
    """
    
    def __call__(self, *args, **kwds):
        referent = _proxied(self)
        return type(referent).__call__(referent, *args, **kwds)
    
    __slots__ = ()
    
    
class ProxyType(AbstractProxyType):
    """ProxyType(obj) -> normal weak proxy to obj.
    
    A normal weak proxy is not hashable no matter what, 
    because of its mutable nature nor is it callable.
    
    Besides these differences it is equivalent to obj,
    including the __class__ attribute. Passing a normal
    weak proxy into type() will still yield the proxy's
    type, not obj's type.
    """  
    
    __slots__ = ()
    
class ReferenceDescriptor(object):
    # Set its doc to an instance of 
    # _reference_descriptor_doc_type
    __doc__ = _reference_descriptor_doc_type()

    def __get__(self, instance, owner=None):
        "Return an attribute of instance, which is of type owner."
        if instance is None:
            return self
        return get_pyweakrefs(instance)        

    __slots__ = ()
    
class ReferenceType(object):
    """ReferenceType(obj) -> weak reference to obj.

    Calling the weak reference will get its object.
    A weak reference does not protect its object from garbage
    collection (unless purging is disabled). 

    type(obj), or a superclass,  must be register()ed. 
    Otherwise, TypeError is raised."""

    def __call__(self):
        "Implement self()."
        # Return self's object
        return _referent(self)
    
    @property
    def __callback__(self):
        return _callback(self)

    def __eq__(self, other):
        "Return self==other"
        if not isinstance(other, ReferenceType):
            return NotImplemented
        o = self()
        if o is None:
            return self is other
        return self() == other()

    def __hash__(self):
        "Return hash(self)."
        # Return the hash value of self's object
        return hash(self())
    
    def __init__(self, obj, callback=None):
        pass

    def __ne__(self):
        "Return self!=other"
        if not isinstance(other, ReferenceType):
            return NotImplemented
        return not (self == other)

    def __new__(cls, obj, callback=None):
        "Create and return a new object.  See help(type) for accurate signature."
        # Check if type(obj) is register()ed
        if not _is_eligible(obj):
            message = "Cannot use pyweakref.ref for class {0.__module__}.{0.__qualname__} instances".format(type(obj))
            raise TypeError(message)
        # Create a new pyweakref...
        self = object.__new__(cls)
        # ...set its object and callback..
        _reference_registry[id(self)] = obj, callback
        # ...add it to the registry...
        _reference_id_registry.setdefault(id(obj), [])
        _reference_id_registry[id(obj)].append(self)
        # ...and return it. Whew!
        return self
    
    def __reduce__(self):
        raise TypeError("cannot pickle pyweakref object")    
    
    def __reduce_ex__(self, protocol):
        raise TypeError("cannot pickle pyweakref object")

    def __repr__(self):
        "Return repr(self)."
        base = f"<pyweakref at 0x{hex(id(self))[2:].upper()}, "
        referent = self()
        if referent is not None:
            return base \
                   + f"to <{type(referent).__module__}.{type(referent).__qualname__} " \
                   + f"object at 0x{hex(id(referent))[2:].upper()}>>"
        else:
            return base + "dead>"

    __slots__ = () 

def circular_reference_count(obj: typing.Any) -> int:
    """Return the number of circular references to the object.
    
    For the purposes of this function, the circular reference
    must be only accessible (directly or indirectly) through the object.
    """
    if np is not None:
        result = _numpy_circular_ref_count(obj)
        if result is not NotImplemented:
            return result
    return _get_circular_ref_count(obj)

def disable_purging() -> None:
    """Disable purging.

    Without purging, pyweakref.ref instances will not be weak references,
    rather strong references. Think twice before calling this function."""
    global _purge, _purge_timer
    if purging():
        _purge = False
        _purge_timer.cancel()

def enable_purging() -> None:
    """Enable purging.

    Only with purging, pyweakref.ref instances will be weak references,
    not strong references."""
    global _purge, _purge_timer
    if not purging():
        _purge = True
        _purge_timer = threading.Timer(5.0, _purge_func)
        _purge_timer.start()
        
def get_pyweakref_count(obj: typing.Any) -> int:
    "Return number of pyweakrefs to obj."
    return len(get_pyweakrefs(obj))
        
def get_pyweakrefs(obj: typing.Any) -> list[ReferenceType]:
    """Return all pyweakrefs to obj. 
    If none, return an empty list."""
    seq = _reference_id_registry.get(id(obj), [])
    return [seq[0] for item in seq]

def purge() -> None:
    """Run a purge cycle right now.

    Automatic purging will not be enabled if currently disabled.
    """
    _purge_func(False)


def purging() -> bool:
    """Return if purging is enabled.

    Without purging, pyweakref.ref instances will not be weak references,
    rather strong references."""
    return _purge

ref = ReferenceType

## Final touches ###
# enable purging
enable_purging()

# Technical names
AbstractProxyType.__name__ = "AbstractProxy"
CallableProxyType.__name__ = "CallableProxy"
ProxyType.__name__ = "Proxy"
ReferenceType.__name__ = "Reference"

# Touching up ProxyType and CallableProxyType
_proxy_builder = {
    "__abs__" : (),
    "__add__" : ('value',),
    "__and__" : ('value',),
    "__bool__": (),
    "__bytes__": (),
    "__contains__": ('key',),
    "__delattr__" : ('name',),
    "__delitem__" : ('key',),
    "__divmod__" : ('value',),
    "__eq__": ('value',),
    "__float__": (),
    "__floordiv__": ('value',),
    "__ge__" : ('value',),
    "__getitem__": ('value',),
    "__gt__": ('value',),
    "__iadd__": ('value',),
    "__iand__": ('value',),
    "__ifloordiv__": ('value',),
    "__ilshift__": ('value',),
    "__imatmul__": ('value',),
    "__imod__": ('value',),
    "__imul__": ('value',),
    "__index__": (),
    "__int__": (),
    "__invert__": (),
    "__ior__": ('value',),
    "__ipow__": ('value',),
    "__irshift__": ('value',),
    "__isub__": ('value',),
    "__iter__": (),
    "__itruediv__": ('value',),
    "__ixor__": ('value',), 
    "__le__": ('value',),
    "__len__": ('value',),
    "__lshift__": ('value',),
    "__lt__": ('value',),
    "__matmul__": ('value',),
    "__mod__" : ('value',),
    "__mul__": ('value',),
    "__ne__": ('value',),
    "__neg__": (),
    "__next__": (),
    "__or__": ('value',),
    "__pos__": (),
    "__radd__": ('value',),
    "__rand__": ('value',),
    "__rdivmod__": ('value',),
    "__reversed__": (),
    "__rfloordiv__": ('value',),
    "__rlshift__": ('value',),
    "__rmatmul__": ('value',),
    "__rmod__": ('value',),
    "__rmul__": ('value',),
    "__ror__": ('value',),
    "__rrshift__": ('value',),
    "__rshift__": ('value',),
    "__rsub__": ('value',),
    "__rtruediv__": ('value',),
    "__rxor__": ('value',), 
    "__setattr__": ('name', 'value'),
    "__setitem__": ('key', 'value'),
    "__str__": (),
    "__sub__": ('value',),
    "__truediv__": ('value',),
    "__xor__": ('value',),
}


for func_name, params in _proxy_builder.items():
    code_text = \
        f"""def funcobj(self, {", ".join(params)}):
                referent = _proxied(self)
                return type(referent).{func_name}(referent, {", ".join(params)})"""
    exec(code_text)
    funcobj.__module__ = "support"
    funcobj.__name__ = func_name
    funcobj.__qualname__ = "AbstractProxyType." + func_name
    setattr(AbstractProxyType, func_name, funcobj)
        
# Clean up
del  _proxy_builder, funcobj, code_text, func_name, params