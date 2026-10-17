from katacr.policy.offline.action_filter import (
    should_skip_for_elixir,
    should_skip_for_unreadable_hand,
    should_execute_with_delay,
)


def test_elixir_filter_enforces_cost_by_default():
    assert should_skip_for_elixir(cost=4, last_elixir=1, enforce=True) is True
    assert should_skip_for_elixir(cost=4, last_elixir=4, enforce=True) is False


def test_elixir_filter_can_be_disabled_for_live_mumu():
    assert should_skip_for_elixir(cost=4, last_elixir=1, enforce=False) is False


def test_unreadable_hand_blocks_taps_but_partial_recognition_does_not():
    assert should_skip_for_unreadable_hand(['empty'] * 5, enforce=True) is True
    assert should_skip_for_unreadable_hand(['empty', 'hog-rider', 'empty', 'empty', 'empty'], enforce=True) is False
    assert should_skip_for_unreadable_hand(['empty'] * 5, enforce=False) is False
    assert should_skip_for_unreadable_hand(None, enforce=True) is False


def test_delay_gate_reports_whether_an_action_can_be_sent():
    assert should_execute_with_delay([1, 4, 17, 8], max_delay=8) is True
    assert should_execute_with_delay([1, 4, 17, 9], max_delay=8) is False
    assert should_execute_with_delay([1, 4, 17, 19], max_delay=None) is True
    assert should_execute_with_delay([0, 4, 17, 0], max_delay=8) is False


if __name__ == "__main__":
    test_elixir_filter_enforces_cost_by_default()
    test_elixir_filter_can_be_disabled_for_live_mumu()
    test_unreadable_hand_blocks_taps_but_partial_recognition_does_not()
    test_delay_gate_reports_whether_an_action_can_be_sent()
    print("action_filter_check passed")
