import os
from cProfile import Profile
from contextlib import contextmanager
from pstats import SortKey, Stats
import signal
from test_rig.run import tf, onnx, cv, tvm

os.environ["USE_GPU_INFERENCE"] = "0"


# https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call

class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def show_profile(recorded_profile: Profile):
    (
        Stats(recorded_profile)
        .strip_dirs()
        .sort_stats(SortKey.TIME)
        .print_stats()
    )


def run_profiled(module):
    with Profile() as profile:
        try:
            with time_limit(10):
                module.run()
        except TimeoutException:
            print("Timed out!")

        show_profile(profile)


if __name__ == "__main__":
    for m in [onnx, cv, tvm]:
        print("Testing {}\n\n".format(m))
        run_profiled(m)
    # tf.run()
