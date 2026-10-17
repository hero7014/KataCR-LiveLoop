def should_skip_for_elixir(cost, last_elixir, enforce=True):
  return bool(enforce and cost > last_elixir)


def should_skip_for_unreadable_hand(raw_cards, enforce=False):
  """Optionally avoid an ADB tap when every raw hand slot is visually unreadable."""
  return bool(enforce and raw_cards) and all(card == 'empty' for card in raw_cards)


def should_execute_with_delay(action, max_delay):
  if not action[0]:
    return False
  return max_delay is None or len(action) == 3 or action[-1] <= max_delay
