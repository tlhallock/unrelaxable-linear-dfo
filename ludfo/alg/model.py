from dataclasses import dataclass

from ludfo.alg.geometry import PoisednessCertification
from ludfo.alg.state import State



@dataclass
class Model:
    success: bool



def udpate_model(state: State, cert: PoisednessCertification) -> Model:
    return Model(
        success=False
    )
