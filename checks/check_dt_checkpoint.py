from pathlib import Path

from katacr.utils.ckpt_manager import CheckpointManager


MODEL = "DT_4L_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step50__0__20240519_224135"


def main():
    path = Path("logs") / "Policy" / MODEL / "ckpt"
    info = CheckpointManager(str(path)).restore(8)
    print("restore_ok", sorted(info.keys()))
    print("config_keys", len(info["config"]), sorted(info["config"].keys())[:8])
    print("has_params", "params" in info["variables"])


if __name__ == "__main__":
    main()
