from dataclasses import dataclass

from ludfo.alg.state import State
from ludfo.alg.model import Model


@dataclass
class CriticalityCheck:
    converged: bool
    critical: bool




def check_criticality(state: State, model: Model) -> CriticalityCheck:
    
    return CriticalityCheck(
        converged=False,
        critical=False
    )