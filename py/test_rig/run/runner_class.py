from abc import ABC, abstractmethod
from typing import Literal
from typing import get_args

DevType = Literal[
    # "cpu",
    "npu",
    # "gpu",
]

class Runner(ABC):
    @abstractmethod
    def run(self, device: DevType, cycles: int):
        pass
