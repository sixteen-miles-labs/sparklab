"""SparkLab's native execution runtime.

The runtime owns model execution, scheduling, distributed coordination, and cache
management. It must not depend on SparkLab's catalog, acquisition, planning, or CLI
control plane.
"""
