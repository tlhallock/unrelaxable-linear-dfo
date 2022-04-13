from dataclasses import dataclass
from alg.state import State

from alg.model import Model


@dataclass
class CriticalityCheck:
    converged: bool
    critical: bool




def check_criticality(state: State, model: Model) -> CriticalityCheck:
    
    return CriticalityCheck(
        converged=False,
        critical=False
    )