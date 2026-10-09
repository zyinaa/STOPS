"""Turn a self-evolving system's run directory into a :class:`stops.trace.Trace`."""

from .gepa import read_gepa
from .skillopt import read_skillopt

__all__ = ["read_skillopt", "read_gepa"]
