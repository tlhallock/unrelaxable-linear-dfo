from dataclasses import dataclass


@dataclass
class Params:
    threshhold_criticality: float = 1e-3
    threshold_tr_radius: float = 1e-3
    threshold_regularity: float = 1e-3
    threshold_reduction_sufficient: float = 1e-3
    threshold_reduction_minimum: float = 1e-3
    tr_update_dec: float = 0.5
    tr_update_inc: float = 1.5
    
    maximum_iterations: int = 1000
