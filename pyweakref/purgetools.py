from .support import (circular_reference_count, disable_purging, enable_purging, purge, purging)
                      
__all__ = ["circular_reference_count", "disable_purging", "enable_purging", "purge", "purging"]

__doc__ = """
Tools to interact with the purger. 

The purger kills orphaned pyweakrefs. When the purger
is disabled, the pyweakrefs are strong references.

A great degree of caution should be excercised when
calling any of this module's functions. Use them for
testing and specialized purposes only."""