import os
import time

os.environ.pop("LD_LIBRARY_PATH", None)

import jax
import jax.numpy as jnp


def main():
    print("backend", jax.default_backend())
    print("devices", jax.devices())
    x = jnp.ones((1024, 1024), dtype=jnp.float32)
    f = jax.jit(lambda a: (a @ a).sum())
    start = time.perf_counter()
    y = f(x).block_until_ready()
    elapsed = time.perf_counter() - start
    print("ok", float(y))
    print("elapsed_sec", f"{elapsed:.4f}")


if __name__ == "__main__":
    main()
