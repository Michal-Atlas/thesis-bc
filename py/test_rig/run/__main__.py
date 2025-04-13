import os
import signal
import typing
from cProfile import Profile
from contextlib import contextmanager
from pstats import SortKey, Stats
import timeit
from typing import Optional, Any, List, Tuple

from test_rig.run.cv_runner import CVRunner
from test_rig.run.onnx_runner import ONNXRunner
from test_rig.run.runner_class import Runner, DevType
from test_rig.run.tf_runner import TFRunner
from test_rig.run.tvm_runner import TVMRunner

os.environ["USE_GPU_INFERENCE"] = "0"

def show_profile(recorded_profile: Profile, file: str):
    with open(file, "w") as f:
        (
            Stats(recorded_profile, stream=f)
            .strip_dirs()
            .sort_stats(SortKey.TIME)
            .print_stats()
        )


def run_profiled(module: Runner):
    devices: Tuple[DevType] = typing.get_args(DevType)
    name = module.__class__.__name__.replace("Runner", "")
    for device in devices:
        print("\n\n=== Testing {} on {} ===\n".format(name, device))
        with Profile() as profile:
            print(timeit.timeit(lambda: module.run(device, 50), number=1))
            show_profile(profile, "{}_{}.log".format(name, device))


if __name__ == "__main__":
    runners: List[Runner] = [
        ONNXRunner(),
        # CVRunner(),
        # TFRunner(),
        # TVMRunner(),
    ]
    for m in runners:
        run_profiled(m)
