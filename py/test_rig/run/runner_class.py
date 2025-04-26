from abc import ABC, abstractmethod
from typing import Literal
from typing import get_args

DevType = Literal[
    "npu",
    "cpu",
    # "gpu",
]

class Runner(ABC):
    def __init__(self, cycles: int, device: DevType):
        self.cycles = cycles
        self.device = device

    @abstractmethod
    def load_data(self):
        pass


    def run(self):
        # for i in range(self.cycles + 1):
        #     print(f"\rCycle {i}/{self.cycles}...", end="")
        self.step()
        # print()

    @abstractmethod
    def step(self):
        pass
