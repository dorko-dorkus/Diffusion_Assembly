"""Namespace package for scheduling utilities.

Allows ``loto.scheduling`` to be extended in separate distributions.
"""

from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
