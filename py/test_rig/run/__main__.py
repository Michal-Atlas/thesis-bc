import os
import signal
import sys
import typing
from cProfile import Profile
from contextlib import contextmanager
from pstats import SortKey, Stats
import timeit
from typing import Optional, Any, List, Tuple
import humanfriendly as hf

import numpy as np

from test_rig.config import CYCLES, MOBILENET_PATH, NPU_DEBUG, NPU_CACHE
from test_rig.run.cv_runner import CVRunner
from test_rig.run.onnx_runner import ONNXRunner
from test_rig.run.runner_class import Runner, DevType
from test_rig.run.tf_runner import TFRunner
from test_rig.run.tvm_runner import TVMRunner

os.environ["USE_GPU_INFERENCE"] = "0"
if NPU_DEBUG:
    os.environ["CNN_PERF"] = "1"
    os.environ["NN_EXT_SHOW_PERF"] = "1"
    os.environ["VIV_VX_DEBUG_LEVEL"] = "1"
    os.environ["VIV_VX_PROFILE"] = "1"
if NPU_CACHE:
    os.environ["VIV_VX_ENABLE_CACHE_GRAPH_BINARY"] = "1"
    os.environ["VIV_VX_CACHE_BINARY_GRAPH_DIR"] = os.curdir


def show_profile(recorded_profile: Profile, file):
    (
        Stats(recorded_profile, stream=file)
        .strip_dirs()
        .sort_stats(SortKey.TIME)
        .print_stats()
    )


def tee(f, s):
    f.write(s)
    print(s, end="")
    sys.stdout.flush()


def run_profiled(name, module, extra):
    devices: Tuple[DevType] = typing.get_args(DevType)
    for device in devices:
        print(f"\n\n=== Testing {name} on {device} ===\n")
        log_file = f"{name}_{device}.log"
        instance = module(
            cycles=CYCLES,
            device=device,
            **extra
        )
        with Profile() as profile:
            with open(log_file, "w") as f:
                time_load_data = timeit.timeit(lambda: instance.load_data(), number=1)
                fmt_load_data = hf.format_timespan(
                    time_load_data,
                    detailed=True,
                )
                tee(f, f"Loadtime: {fmt_load_data}\n")
                for i in range(CYCLES):
                    instance.load_data()
                    time_run = timeit.timeit(lambda: instance.run(), number=1)
                    time_fmt = hf.format_timespan(
                        time_run,
                        detailed=True,
                    )
                    tee(f, f"Runtime #{i}: {time_fmt}\n")

                show_profile(profile, f)


if __name__ == "__main__":
    # torch.set_default_dtype(torch.uint8)
    runners = [
        # ("ONNX", ONNXRunner, {}),
        # ("OpenCV", CVRunner, {}),
        ("TensorFlow", TFRunner, {}),
        ("TensorFlowMobileNet", TFRunner, {
            "model_path": MOBILENET_PATH,
            "dtype": np.uint8,
        }),
        # TVMRunner,
    ]
    for (n, m, e) in runners:
        run_profiled(n, m, e)
