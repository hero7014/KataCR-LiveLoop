from collections import Counter


def cards_hidden_by_smoothing(raw_cards, cards_memory):
  deploy_cards = set()
  for card, memory in zip(raw_cards, cards_memory):
    if card == 'empty' and memory not in ('empty', None):
      deploy_cards.add(memory)
  return deploy_cards


def should_drop_duplicate_frame(cards, live_smoothing=False):
  counts = Counter(c for c in cards if c != 'empty')
  return (not live_smoothing) and any(count > 1 for count in counts.values())


def should_remember_unmatched_elixir(live_smoothing=False):
  """Deduplicate a confirmed elixir track when live card OCR has no match."""
  return live_smoothing


class CardStateSmoother:
  def __init__(self, empty_ttl=5):
    self.empty_ttl = empty_ttl
    self.reset()

  def reset(self):
    self.memory = [None] * 5
    self.empty_age = [0] * 5

  def update(self, cards):
    cards = list(cards)
    counts = Counter(c for c in cards if c != 'empty')
    smoothed = list(cards)
    duplicate_cards = {card for card, count in counts.items() if count > 1}

    for i, card in enumerate(cards):
      memory = self.memory[i]
      if card == 'empty':
        self.empty_age[i] += 1
        if (
            memory not in (None, 'empty')
            and self.empty_age[i] <= self.empty_ttl
            and counts[memory] == 0
        ):
          smoothed[i] = memory
        continue

      self.empty_age[i] = 0
      if (
          card in duplicate_cards
          and memory not in (None, 'empty')
          and memory != card
          and counts[memory] == 0
      ):
        smoothed[i] = memory

    for i, card in enumerate(smoothed):
      if card != 'empty':
        self.memory[i] = card
      elif self.empty_age[i] > self.empty_ttl:
        self.memory[i] = card

    return smoothed
