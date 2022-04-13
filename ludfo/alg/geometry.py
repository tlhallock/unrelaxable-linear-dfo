from dataclasses import dataclass
from alg.state import State

from alg.model import Model


@dataclass
class PoisednessCertification:
    converged: bool
    critical: bool




def improve_geometry(state: State) -> PoisednessCertification:
    
    return PoisednessCertification(
        converged=False,
        critical=False
    )



