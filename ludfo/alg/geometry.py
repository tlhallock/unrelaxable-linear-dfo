from dataclasses import dataclass
from ludfo.alg.state import State

from ludfo.alg.model import Model


@dataclass
class PoisednessCertification:
    converged: bool
    critical: bool




def improve_geometry(state: State) -> PoisednessCertification:
    
    return PoisednessCertification(
        converged=False,
        critical=False
    )



