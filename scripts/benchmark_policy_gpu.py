import argparse
import os
import sys
import time
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ.pop("LD_LIBRARY_PATH", None)

path_root = Path(__file__).resolve().parents[1]
sys.path.append(str(path_root))

import jax
import numpy as np

from katacr.policy.offline.eval_no_delay import MODEL_NAME
from katacr.policy.offline.starformer_no_delay import StARConfig, StARformer, TrainConfig
from katacr.utils.ckpt_manager import CheckpointManager


def load_policy(model_name: str, load_epoch: int):
    path_weights = path_root / "logs" / "Policy" / model_name / "ckpt"
    ckpt_mngr = CheckpointManager(str(path_weights))
    load_info = ckpt_mngr.restore(load_epoch)
    params, cfg = load_info["variables"]["params"], load_info["config"]
    if "cnn_mode" not in cfg:
        cfg["cnn_mode"] = "resnet"
    model = StARformer(StARConfig(**cfg))
    model.create_fns()
    state = model.get_state(TrainConfig(**cfg), train=False)
    state = state.replace(params=params, tx=None, opt_state=None)
    return model, state


def build_inputs(model):
    batch = 1
    steps = model.cfg.n_step
    bars = model.cfg.n_bar_size
    s = {
        "arena": np.zeros((batch, steps, 32, 18, 1 + 1 + bars * 2), np.int32),
        "arena_mask": np.zeros((batch, steps, 32, 18), np.bool_),
        "cards": np.zeros((batch, steps, 5), np.int32),
        "elixir": np.zeros((batch, steps), np.int32),
    }
    a = {
        "select": np.zeros((batch, steps), np.int32),
        "pos": np.zeros((batch, steps, 2), np.int32),
    }
    rtg = np.ones((batch, steps), np.float32)
    timestep = np.arange(steps, dtype=np.int32).reshape(batch, steps)
    return s, a, rtg, timestep, steps


def timed_predict(model, state, inputs, rng, deterministic):
    s, a, rtg, timestep, step_len = inputs
    start = time.perf_counter()
    out = model.predict(state, s, a, rtg, timestep, step_len, rng, deterministic)
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out)
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--load-epoch", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    print("backend", jax.default_backend())
    print("devices", jax.devices())
    model, state = load_policy(args.model_name, args.load_epoch)
    inputs = build_inputs(model)
    rng = jax.random.PRNGKey(42)

    compile_sec = timed_predict(model, state, inputs, rng, args.deterministic)
    print("compile_plus_first_sec", f"{compile_sec:.4f}")

    times = []
    for i in range(args.runs):
        rng, step_rng = jax.random.split(rng)
        elapsed = timed_predict(model, state, inputs, step_rng, args.deterministic)
        times.append(elapsed)
        print(f"run_{i + 1}_sec", f"{elapsed:.4f}")
    print("steady_avg_sec", f"{sum(times) / len(times):.4f}")


if __name__ == "__main__":
    main()
