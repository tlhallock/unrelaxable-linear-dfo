from dataclasses import dataclass



# Are these two classes the same?
@dataclass
class IterationResult:
    success: bool
    converged: bool
    completed: bool
    message: str
    
    
@dataclass
class RunResult:
    success: bool
    converged: bool
    message: str
    
    @staticmethod
    def from_iteration(iteration_result):
        return RunResult(**iteration_result.dict())

