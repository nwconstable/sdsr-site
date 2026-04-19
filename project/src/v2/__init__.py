"""V2 mosquito-risk mapping workflow for the SDSR repository.

This package is the namespace boundary for the parallel v2 benchmark. New
benchmark-specific logic for sampled wetland windows, latent mosquito-risk
fields, free-movement drone simulation, v2 training loops, and v2 evaluation
belongs here rather than being retrofitted into the v1 shortest-path modules.

V2 should reuse generic repository utilities where their semantics still fit,
especially `load_data.py` for GeoPackage access and `comms.py` for
communication scheduling, dropout, and logging.
"""
