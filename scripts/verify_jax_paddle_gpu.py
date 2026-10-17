import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import jax
import jax.numpy as jnp
import paddle


def main():
  print("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", ""))
  print("jax_backend", jax.default_backend())
  print("jax_devices", jax.devices())
  f = jax.jit(lambda x: (x @ x).sum())
  x = jnp.ones((512, 512), dtype=jnp.float32)
  y = f(x).block_until_ready()
  print("jax_ok", float(y))

  print("paddle", paddle.__version__)
  print("paddle_compiled_cuda", paddle.device.is_compiled_with_cuda())
  paddle.set_device("gpu:0")
  px = paddle.ones([512, 512], dtype="float32")
  py = paddle.matmul(px, px)
  print("paddle_place", py.place)
  print("paddle_ok", float(py[0, 0]))


if __name__ == "__main__":
  main()
