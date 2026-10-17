import unittest

import numpy as np

from scripts.adb_ui_flow import (
    PageState,
    detect_page_state,
    resolve_action,
)


def blank():
    return np.zeros((1280, 720, 3), dtype=np.uint8)


def fill_rect(img, x1, y1, x2, y2, bgr):
    img[y1:y2, x1:x2] = np.array(bgr, dtype=np.uint8)
    return img


class AdbUiFlowTest(unittest.TestCase):
    def test_detects_home_from_battle_button_region(self):
        img = blank()
        fill_rect(img, 250, 920, 485, 1050, (0, 185, 255))

        self.assertEqual(detect_page_state(img), PageState.HOME)

    def test_detects_post_match_from_confirm_button_region(self):
        img = blank()
        fill_rect(img, 275, 1110, 455, 1215, (255, 170, 60))

        self.assertEqual(detect_page_state(img), PageState.POST_MATCH)

    def test_unknown_page_does_not_allow_next_action(self):
        decision = resolve_action("next", PageState.UNKNOWN, confirmed=True)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.action, "none")

    def test_suggest_next_on_home_does_not_execute_without_confirmation(self):
        decision = resolve_action("next", PageState.HOME, confirmed=False)

        self.assertEqual(decision.action, "battle")
        self.assertFalse(decision.allowed)

    def test_confirmed_next_on_home_allows_battle(self):
        decision = resolve_action("next", PageState.HOME, confirmed=True)

        self.assertEqual(decision.action, "battle")
        self.assertTrue(decision.allowed)

    def test_force_next_on_home_resolves_to_real_action(self):
        decision = resolve_action("next", PageState.HOME, force=True)

        self.assertEqual(decision.action, "battle")
        self.assertTrue(decision.allowed)

    def test_confirm_command_only_allowed_on_post_match(self):
        home_decision = resolve_action("confirm", PageState.HOME, confirmed=True)
        post_match_decision = resolve_action("confirm", PageState.POST_MATCH, confirmed=True)

        self.assertFalse(home_decision.allowed)
        self.assertTrue(post_match_decision.allowed)


if __name__ == "__main__":
    unittest.main()
