import numpy as np

from scripts.adb_card_probe import extract_card_crops, make_contact_sheet, normalize_live_frame


def test_live_normalization_matches_supported_ratio():
  frame = np.zeros((1280, 720, 3), np.uint8)
  normalized = normalize_live_frame(frame)
  assert normalized.shape == (1280, 592, 3)


def test_card_crops_follow_part3_geometry():
  frame = np.zeros((1280, 592, 3), np.uint8)
  part3, crops = extract_card_crops(frame)
  assert part3.shape == (200, 600, 3)
  assert len(crops) == 5
  assert all(crop.size > 0 for crop in crops)


def test_contact_sheet_contains_each_card_tile():
  crops = [np.zeros((50, 30, 3), np.uint8) for _ in range(5)]
  sheet = make_contact_sheet(crops, ["a", "b", "c", "d", "e"], tile_size=(40, 60))
  assert sheet.shape == (60, 200, 3)


if __name__ == "__main__":
  test_live_normalization_matches_supported_ratio()
  test_card_crops_follow_part3_geometry()
  test_contact_sheet_contains_each_card_tile()
  print("adb_card_probe_check passed")
