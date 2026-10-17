from pathlib import Path


EVAL_PATH = Path(__file__).parents[1] / "katacr" / "policy" / "offline" / "eval.py"
RUNNER_PATH = Path(__file__).parents[1] / "scripts" / "run_mumu_dt_eval.sh"
SAR_BUILDER_PATH = Path(__file__).parents[1] / "katacr" / "policy" / "perceptron" / "sar_builder.py"
SAR_DAEMON_PATH = Path(__file__).parents[1] / "katacr" / "policy" / "env" / "sar_daemon.py"


def test_live_dt_parameters_are_explicit():
  source = EVAL_PATH.read_text(encoding="utf-8")
  assert 'parser.add_argument("--rtg", type=float, default=3.0' in source
  assert 'parser.add_argument("--save-decision-trace"' in source
  assert 'parser.add_argument("--max-action-delay", type=int, default=8' in source
  assert 'rtg=args.rtg' in source
  assert 'save_decision_trace=args.save_decision_trace' in source
  assert 'max_action_delay=args.max_action_delay' in source
  assert "'visual_cards_before_action': previous_visual_cards" in source
  assert "'visual_cards_after_step': info.get('raw_cards')" in source


def test_runner_forwards_rtg_and_trace_setting():
  source = RUNNER_PATH.read_text(encoding="utf-8")
  assert 'BASE_RTG="${KATACR_BASE_RTG:-3.0}"' in source
  assert '--rtg "$BASE_RTG"' in source
  assert 'MAX_ACTION_DELAY="${KATACR_MAX_ACTION_DELAY:-8}"' in source
  assert '--max-action-delay "$MAX_ACTION_DELAY"' in source
  assert '--save-decision-trace "${KATACR_SAVE_DECISION_TRACE:-True}"' in source


def test_trace_preserves_raw_visual_hand_before_smoothing():
  builder = SAR_BUILDER_PATH.read_text(encoding="utf-8")
  daemon = SAR_DAEMON_PATH.read_text(encoding="utf-8")
  assert "self.last_raw_cards = list(info['cards'])" in builder
  assert "info['raw_cards'] = self.sar_builder.last_raw_cards" in daemon


if __name__ == "__main__":
  test_live_dt_parameters_are_explicit()
  test_runner_forwards_rtg_and_trace_setting()
  test_trace_preserves_raw_visual_hand_before_smoothing()
  print("dt_eval_cli_check passed")
