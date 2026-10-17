from katacr.policy.env.utils import resolve_tap_xy


def main():
  assert resolve_tap_xy((0.5, 0.25), relative=True, img_size=(592, 1280), scale=(720 / 592, 1)) == (360, 320)
  assert resolve_tap_xy((226.2, 1108.7), relative=False) == (226, 1108)
  assert resolve_tap_xy((19, 901), relative=False, clamp=(70, 260, 650, 930)) == (70, 901)
  assert resolve_tap_xy((700, 120), relative=False, clamp=(70, 260, 650, 930)) == (650, 260)


if __name__ == "__main__":
  main()
