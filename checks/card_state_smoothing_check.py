from katacr.policy.perceptron.card_state_smoother import (
    CardStateSmoother,
    cards_hidden_by_smoothing,
    should_drop_duplicate_frame,
    should_remember_unmatched_elixir,
)


def test_empty_slots_reuse_recent_stable_cards():
    smoother = CardStateSmoother()
    smoother.update(["hog-rider", "cannon", "skeletons", "fireball", "the-log"])

    cards = smoother.update(["hog-rider", "empty", "skeletons", "fireball", "the-log"])

    assert cards == ["hog-rider", "cannon", "skeletons", "fireball", "the-log"]


def test_duplicate_non_empty_cards_do_not_discard_whole_frame():
    smoother = CardStateSmoother()
    smoother.update(["hog-rider", "cannon", "skeletons", "fireball", "the-log"])

    cards = smoother.update(["hog-rider", "cannon", "cannon", "fireball", "the-log"])

    assert cards == ["hog-rider", "cannon", "skeletons", "fireball", "the-log"]


def test_empty_slot_does_not_reuse_card_already_visible_elsewhere():
    smoother = CardStateSmoother()
    smoother.update(["cannon", "ice-golem", "the-log", "hog-rider", "skeletons"])

    cards = smoother.update(["empty", "ice-golem", "the-log", "cannon", "skeletons"])

    assert cards == ["empty", "ice-golem", "the-log", "cannon", "skeletons"]


def test_duplicate_fix_does_not_reuse_card_already_visible_elsewhere():
    smoother = CardStateSmoother()
    smoother.update(["ice-spirit", "fireball", "hog-rider", "cannon", "ice-golem"])

    cards = smoother.update(["hog-rider", "fireball", "hog-rider", "cannon", "ice-spirit"])

    assert cards == ["hog-rider", "fireball", "hog-rider", "cannon", "ice-spirit"]


def test_raw_empty_still_marks_deploy_card_after_smoothing():
    raw_cards = ["hog-rider", "empty", "skeletons", "ice-golem", "cannon"]
    memory = ["hog-rider", "the-log", "skeletons", "ice-golem", "cannon"]

    assert cards_hidden_by_smoothing(raw_cards, memory) == {"the-log"}


def test_live_smoothing_does_not_drop_duplicate_frame():
    cards = ["hog-rider", "ice-spirit", "hog-rider", "fireball", "the-log"]

    assert should_drop_duplicate_frame(cards, live_smoothing=False) is True
    assert should_drop_duplicate_frame(cards, live_smoothing=True) is False


def test_live_mode_remembers_unmatched_elixir_track():
    assert should_remember_unmatched_elixir(live_smoothing=True) is True
    assert should_remember_unmatched_elixir(live_smoothing=False) is False


if __name__ == "__main__":
    test_empty_slots_reuse_recent_stable_cards()
    test_duplicate_non_empty_cards_do_not_discard_whole_frame()
    test_empty_slot_does_not_reuse_card_already_visible_elsewhere()
    test_duplicate_fix_does_not_reuse_card_already_visible_elsewhere()
    test_raw_empty_still_marks_deploy_card_after_smoothing()
    test_live_smoothing_does_not_drop_duplicate_frame()
    test_live_mode_remembers_unmatched_elixir_track()
    print("card_state_smoothing_check passed")
