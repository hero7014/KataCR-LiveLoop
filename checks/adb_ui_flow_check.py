import numpy as np

from scripts.adb_ui_flow import PageState, detect_page_state, detect_page_signals


def test_home_uses_battle_button_signal():
    image = np.zeros((1280, 720, 3), dtype=np.uint8)
    image[895:1080, 235:500] = (50, 150, 220)

    assert detect_page_state(image) == PageState.HOME


def test_post_match_requires_blue_confirm_and_dimmed_backdrop():
    image = np.full((1280, 720, 3), 60, dtype=np.uint8)
    image[1095:1235, 260:470] = (200, 120, 80)

    signals = detect_page_signals(image)

    assert signals["confirm_blue"] > 0.18
    assert signals["backdrop_dim"] > 0.20
    assert detect_page_state(image) == PageState.POST_MATCH


def test_bright_battle_card_bar_is_not_post_match():
    image = np.full((1280, 720, 3), (140, 190, 220), dtype=np.uint8)
    image[1095:1235, 260:470] = (200, 120, 80)

    assert detect_page_state(image) == PageState.UNKNOWN


if __name__ == "__main__":
    test_home_uses_battle_button_signal()
    test_post_match_requires_blue_confirm_and_dimmed_backdrop()
    test_bright_battle_card_bar_is_not_post_match()
    print("adb_ui_flow_check passed")
