from dataclasses import dataclass
from alg.geometry import PoisednessCertification

from alg.state import State



@dataclass
class Model:
    success: bool



def udpate_model(state: State, cert: PoisednessCertification) -> Model:
    return Model(
        success=False
    )
