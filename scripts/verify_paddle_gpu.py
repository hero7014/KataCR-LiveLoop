import paddle


def main():
  print("paddle", paddle.__version__)
  print("compiled_cuda", paddle.device.is_compiled_with_cuda())
  print("device_before", paddle.device.get_device())
  paddle.set_device("gpu:0")
  print("device_after", paddle.device.get_device())
  x = paddle.ones([1024, 1024], dtype="float32")
  y = paddle.matmul(x, x)
  print("tensor_place", y.place)
  print("ok", float(y[0, 0]))


if __name__ == "__main__":
  main()
