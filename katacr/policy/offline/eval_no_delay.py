"""
Open phone screen video stream:
sudo modprobe v4l2loopback
scrcpy --v4l2-sink=/dev/video2 --no-video-playback
"""
import os, sys
from pathlib import Path
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
path_root = Path(__file__).parents[3]
sys.path.append(str(path_root))
from katacr.utils.ckpt_manager import CheckpointManager
from katacr.policy.offline.dataset import build_feature
import cv2, jax
from katacr.policy.env.interact_env import InteractEnv
from katacr.policy.env.video_env import VideoEnv
import numpy as np
from katacr.utils import colorstr, Stopwatch
from katacr.utils.ffmpeg.merge_videos import merge_videos_left_and_right
from katacr.constants.card_list import card2elixir
from katacr.policy.replay_data.data_display import GridDrawer
import json, time
from katacr.utils.csv_writer import CSVWriter

MODEL_NAME = "StARformer_no_delay_2L_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step50__0__20240520_205252"
CSV_TITLE = ['episode', 'model_name', 'epoch', 'survival_time', 'total_reward', 'use_actions']
class_merge = [
  {'ice-spirit', 'ice-spirit-evolution'},
  {'skeleton', 'skeleton-evolution'}
]

def str2bool(x):
  return str(x).lower() in ['1', 'true', 'yes', 'y']

def sanitize_output_name(name: str) -> str:
  name = name.rstrip(' .')
  return name or "video"

def build_eval_output_dir(vid_path=None):
  path = path_root / "logs" / "video_eval" / time.strftime("%Y%m%d_%H%M%S")
  if vid_path is not None:
    path = path / sanitize_output_name(Path(vid_path).stem)
  path.mkdir(parents=True, exist_ok=True)
  return path

def json_safe(value):
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, dict):
    return {str(k): json_safe(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [json_safe(v) for v in value]
  return value

class JSONLWriter:
  def __init__(self, path):
    self.path = Path(path)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self.f = self.path.open('w', encoding='utf-8')

  def write(self, row):
    self.f.write(json.dumps(json_safe(row), ensure_ascii=False, separators=(',', ':')) + '\n')
    self.f.flush()

  def close(self):
    if not self.f.closed:
      self.f.close()

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:
  """ This function would pad at the end of certain axis, https://stackoverflow.com/a/49766444 """
  pad_size = target_length - array.shape[axis]
  if pad_size <= 0:
    return array
  npad = [(0, 0)] * array.ndim
  npad[axis] = (0, pad_size)
  return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

class Evaluator:
  s_key = ['arena', 'arena_mask', 'cards', 'elixir']
  a_key = ['select', 'pos']

  def __init__(
      self, path_weights, load_epoch=None, vid_path=None, show=True, save=False,
      rtg=3., deterministic=True, verbose=False, show_predict=True, eval_num=None,
      fixed_window=True, reward_ocr_interval=10,
      time_ocr_interval=3, elixir_ocr_interval=2,
      save_video=False, save_video_interval=10,
      save_frame_log=True
    ):
    self.base_rtg, self.deterministic = rtg, deterministic
    self.verbose, self.show_predict, self.eval_num = verbose, show_predict, eval_num
    self.fixed_window = fixed_window
    self.show, self.save = show, save
    self.save_frame_log = save_frame_log
    if vid_path is not None:
      self.path_save_dir = build_eval_output_dir(vid_path)
      self.env = VideoEnv(
        vid_path, action_freq=2, show=show, save=save, verbose=verbose,
        reward_ocr_interval=reward_ocr_interval,
        time_ocr_interval=time_ocr_interval,
        elixir_ocr_interval=elixir_ocr_interval,
        path_save_dir=self.path_save_dir,
        save_video=save_video,
        save_video_interval=save_video_interval)
    else:
      self.env = InteractEnv(show=show, save=save)
      self.path_save_dir = self.env.path_save_dir
    self.rng = jax.random.PRNGKey(42)
    self.open_window = False
    self._load_model(path_weights, load_epoch)
    self.vid_writer = None
    self.csv_writer = CSVWriter(self.path_save_dir / f"{self.model_name}_load{self.load_epoch}.csv", title=CSV_TITLE)
    self.frame_log_writer = None
    if self.save and self.save_frame_log:
      self.frame_log_writer = JSONLWriter(self.path_save_dir / f"{self.model_name}_load{self.load_epoch}_frames.jsonl")

  def _load_model(self, path_weights, load_epoch=None):
    print("Loading policy model...", end='')
    ckpt_mngr = CheckpointManager(str(path_weights))
    self.model_name = Path(path_weights).parent.name
    if load_epoch is not None:
      self.load_epoch = load_epoch
    else:
      self.load_epoch = int(sorted(Path(path_weights).glob('*'))[-1].name)
    load_info = ckpt_mngr.restore(self.load_epoch)
    params, cfg = load_info['variables']['params'], load_info['config']
    if 'starformer_no_delay' in str(path_weights).lower():
      from katacr.policy.offline.starformer_no_delay import StARformer, StARConfig, TrainConfig
      if 'cnn_mode' not in cfg:
        cfg['cnn_mode'] = 'resnet'
      self.model = StARformer(StARConfig(**cfg))
    if 'ViDformer' in str(path_weights):
      from katacr.policy.offline.vidformer import ViDformer, ViDConfig, TrainConfig
      self.model = ViDformer(ViDConfig(**cfg))
      self.model_name = 'ViDformer'
    self.model.create_fns()
    state = self.model.get_state(TrainConfig(**cfg), train=False)
    self.state = state.replace(params=params, tx=None, opt_state=None)
    self.n_step = self.model.cfg.n_step
    self.n_bar_size = self.model.cfg.n_bar_size
    self._warmup()
    self.sw = [Stopwatch() for _ in range(2)]
    self.idx2card = self.env.idx2card
    print("Complete!")
  
  def _init_sart(self):
    self.s = {k: [] for k in self.s_key}
    self.a = {k: [] for k in self.a_key}
    self.rtg = []
    self.timestep = []
  
  def _add_sart(self, s, a, rtg, timestep):
    ns, na = build_feature(s, a)
    for x, nx in zip([self.s, self.a], [ns, na]):
      for k in x.keys():
        x[k].append(nx[k])
    self.rtg.append(np.array(rtg, np.float32))
    timestep = min(timestep, self.model.cfg.max_timestep)
    self.timestep.append(np.array(timestep, np.int32))
  
  def _warmup(self):
    B, l = 1, self.n_step
    s = {
      'arena': np.empty((B, l, 32, 18, 1+1+self.n_bar_size*2), np.int32),
      'arena_mask': np.empty((B, l, 32, 18), np.bool_),
      'cards': np.empty((B, l, 5,), np.int32),
      'elixir': np.empty((B, l), np.int32),
    }
    a = {
      'select': np.empty((B, l), np.int32),
      'pos': np.empty((B, l, 2), np.int32)
    }
    r, timestep = np.empty((B, l), np.float32), np.empty((B, l), np.int32)
    jax.device_get(self.model.predict(
      self.state,
      s, a, r, timestep, np.array([l], np.int32), self.rng, self.deterministic))

  def _pad_history(self, values, n_step):
    recent = np.stack(values[-n_step:])
    if recent.shape[0] == n_step:
      return np.expand_dims(recent, 0)
    padded = np.zeros((n_step, *recent.shape[1:]), dtype=recent.dtype)
    if self.fixed_window:
      padded[-recent.shape[0]:] = recent
    else:
      padded[:recent.shape[0]] = recent
    return np.expand_dims(padded, 0)
  
  def get_action(self):
    n_step = self.n_step
    def pad(x):
      return self._pad_history(x, n_step)
    step_len = n_step if self.fixed_window else min(len(self.timestep), n_step)
    step_len_arg = np.array([step_len], np.int32)
    rng, self.rng = jax.random.split(self.rng)
    # for k, v in self.s.items():
    #   print(f"{k}: {pad(v).shape}")
    # for k, v in self.a.items():
    #   print(f"{k}: {pad(v).shape}")
    # print(pad(self.rtg).shape, pad(self.timestep).shape, step_len, rng, self.deterministic)
    profile = {}
    t = time.perf_counter()
    s = {k: pad(v) for k, v in self.s.items()}
    a = {k: pad(v) for k, v in self.a.items()}
    rtg = pad(self.rtg)
    timestep = pad(self.timestep)
    profile['build'] = time.perf_counter() - t
    t = time.perf_counter()
    predict_out = self.model.predict(
      self.state,
      s, a, rtg, timestep,
      step_len_arg, rng, self.deterministic)
    profile['dispatch'] = time.perf_counter() - t
    t = time.perf_counter()
    action, logits_select, logits_pos = jax.device_get(predict_out)
    profile['device_get'] = time.perf_counter() - t
    t = time.perf_counter()
    action = action[0]
    prob_select = np.exp(logits_select)[0]
    prob_select /= prob_select.sum()
    prob_pos = np.exp(logits_pos-logits_pos.max())[0].reshape(32, 18)
    prob_pos /= prob_pos.sum()
    # if step_len == 30:
    #   np.save("/home/yy/Coding/GitHub/KataCR/logs/intercation/video1_eval_dataset_50.npy", data, allow_pickle=True)
    #   exit()
    prob_img = None
    if self.show_predict:
      sel_drawer = GridDrawer(1, 5, size=(576, 50))
      for i in range(5):
        prob = prob_select[i]
        sel_drawer.paint((i, 0), (0, 0, int(255*prob)), f"{prob*100:.2f}")
      simg = np.array(sel_drawer.image)
      # if not self.open_window:
      #   cv2.namedWindow("Predict Select Probability", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
      #   cv2.resizeWindow("Predict Select Probability", img.shape[:2][::-1])
      # cv2.imshow("Predict Select Probability", img)
      # cv2.waitKey(1)
      pos_drawer = GridDrawer(32, 18, size=(576, 846))
      for i in range(32):
        for j in range(18):
          prob = prob_pos[i,j]
          pos_drawer.paint((j, i), (0,0,int(255*prob)), f"{prob*100:.1f}")
      pimg = np.array(pos_drawer.image)
      prob_img = np.concatenate([pimg, simg], 0)
      # if not self.open_window:
      #   self.open_window = True
      #   cv2.namedWindow("Predict Probability", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
      #   cv2.resizeWindow("Predict Probability", img.shape[:2][::-1])
      # cv2.imshow("Predict Probability", img)
      # cv2.waitKey(1)
      # if self.save:
      #   if self.vid_writer is None:
      #     self.path_save_vid = self.path_save_dir / f"{self.episode}_predict.mp4"
      #     self.vid_writer = cv2.VideoWriter(str(self.path_save_vid), cv2.VideoWriter_fourcc(*'mp4v'), 10, (576, 896))
      #   self.vid_writer.write(img)
    profile['postprocess'] = time.perf_counter() - t
    if self.verbose:
      print(colorstr("Time used (Policy detail):"), *[f"{k}={v*1e3:.1f}ms" for k, v in profile.items()])
    return action, prob_img
  
  # def _init_vid_writer(self):
  #   if self.path_save_dir is None: return
  #   if self.vid_writer is not None:
  #     time.sleep(3)
  #     self.vid_writer.release()
  #     path_detection_vid = self.path_save_vid.with_stem(str(self.episode))
  #     path_origin_vid = self.path_save_vid.with_stem(str(self.episode)+'_org')
  #     path_merge_vid = merge_videos_left_and_right(path_detection_vid, self.path_save_vid)
  #     merge_videos_left_and_right(path_origin_vid, path_merge_vid)
  #     for p in [path_detection_vid, self.path_save_vid, path_origin_vid, path_merge_vid]:
  #       p.unlink()
  #     self.vid_writer = None
  
  def eval(self):
    self.episode = 0
    while True:
      scores, use_actions = [], 0
      self._init_sart()
      s, a, _ = self.env.reset(auto=self.eval_num is not None)
      self.episode += 1
      last_elixir = 0
      now_rtg, done = self.base_rtg, False
      self._add_sart(s, a, self.base_rtg, s['time'])
      while not done:
        prev_s = s
        if s['elixir'] is not None: last_elixir = s['elixir']
        with self.sw[0]:
          policy_action, prob_img = self.get_action()
        policy_action = np.array(policy_action)
        raw_policy_action = policy_action.copy()
        card = self.idx2card[str(s['cards'][policy_action[0]])]
        if policy_action[0] and card == 'empty':
          print(f"Skip action, since card index {policy_action[0]} is 'empty'")
          policy_action[0] = 0  # Skip
        if policy_action[0] and card2elixir[card] > last_elixir:
          print(f"Skip action, since no enough elixir for card {card}={card2elixir[card]} > {last_elixir}")
          policy_action[0] = 0  # Skip
        with self.sw[1]:
          s, target_action, r, done, info = self.env.step(policy_action, max_delay=None, prob_img=prob_img)
          # s, a, r, done = self.env.step(a)
        if policy_action[0]: use_actions += 1
        a = {'card_id': policy_action[0], 'xy': policy_action[1:3] if policy_action[0] != 0 else None}
        if self.frame_log_writer is not None:
          self.frame_log_writer.write({
            'episode': self.episode,
            'step_index': len(scores),
            'visual_count': info.get('visual_count'),
            'source_frame': info.get('source_frame'),
            'total_frame': info.get('total_frame'),
            'prev_state': {
              'time': prev_s.get('time'),
              'elixir': prev_s.get('elixir'),
              'cards': prev_s.get('cards'),
            },
            'state': {
              'time': s.get('time'),
              'elixir': s.get('elixir'),
              'cards': s.get('cards'),
            },
            'policy_action_raw': {
              'card_id': raw_policy_action[0],
              'xy': raw_policy_action[1:3],
              'card_name': card,
            },
            'policy_action': a,
            'target_action': target_action,
            'reward': r,
            'total_reward': info.get('total_reward'),
            'done': done,
            'timings': {
              'policy': self.sw[0].dt,
              'step': self.sw[1].dt,
              'env': info.get('timings', {}),
            },
          })
        # print("Action:", a)
        # When use future action predict, a[0] in [1,2,3,4]
        if self.verbose:
          print(colorstr("Time used (Eval):"), *[f"{k}={self.sw[i].dt*1e3:.1f}ms" for k, i in zip(['policy', 'step'], range(2))])
        now_rtg = max(now_rtg-r, 1)
        self._add_sart(s, a, now_rtg, s['time'])
        score = info['total_reward']
        scores.append(score)
      if hasattr(self.env, 'close'):
        self.env.close()
      print(f"score {score}, timestep {s['time']}")
      import matplotlib.pyplot as plt
      plt.plot(scores)
      plt.tight_layout()
      plt.savefig(str(self.path_save_dir / f"scores_{self.episode}.png"))
      plt.close()
      self.csv_writer.write([self.episode, self.model_name, self.load_epoch, s['time'], score, use_actions])
      # self._init_vid_writer()
      if self.eval_num is not None and self.episode == self.eval_num:
        break
    if self.frame_log_writer is not None:
      self.frame_log_writer.close()
    print(colorstr("Finish all evaluation!"))

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-name", default=MODEL_NAME,
    help="The policy model weights directory name in 'KataCR/logs/Policy/{model_name}'")
  parser.add_argument("--load-epoch", type=int, default=None,
    help="The load epoch id in 'KataCR/logs/Policy/{model_name}/ckpt/{load_epoch}'")
  parser.add_argument("--eval-num", type=int, default=20,
    help="The automatically evaluation number times")
  parser.add_argument("--vid-path", type=str, default=None,
    help="If given, evaluate on an offline video instead of interacting with the phone")
  parser.add_argument("--show", default=True, const=True, nargs='?', type=str2bool,
    help="Whether to display the processed video window")
  parser.add_argument("--save", default=True, const=True, nargs='?', type=str2bool,
    help="Whether to save evaluation outputs")
  parser.add_argument("--deterministic", default=True, const=True, nargs='?', type=str2bool,
    help="Whether to use deterministic policy inference")
  parser.add_argument("--verbose", default=True, const=True, nargs='?', type=str2bool,
    help="Whether to print timing information")
  parser.add_argument("--show-predict", default=True, const=True, nargs='?', type=str2bool,
    help="Whether to build and display the policy probability image")
  parser.add_argument("--fixed-window", default=True, const=True, nargs='?', type=str2bool,
    help="Whether to left-pad history and always infer from the fixed final timestep")
  parser.add_argument("--reward-ocr-interval", type=int, default=10,
    help="Run tower HP reward OCR every N visual frames in offline video mode")
  parser.add_argument("--time-ocr-interval", type=int, default=3,
    help="Run battle time OCR every N visual frames in offline video mode")
  parser.add_argument("--elixir-ocr-interval", type=int, default=2,
    help="Run elixir OCR every N visual frames in offline video mode")
  parser.add_argument("--save-video", default=False, const=True, nargs='?', type=str2bool,
    help="Whether to save an annotated detection preview video")
  parser.add_argument("--save-video-interval", type=int, default=10,
    help="Write one annotated video frame every N visual frames")
  parser.add_argument("--save-frame-log", default=True, const=True, nargs='?', type=str2bool,
    help="Whether to save per-decision JSONL records with state, action, reward, and timings")
  args = parser.parse_args()
  path_weights = path_root / f"logs/Policy/{args.model_name}/ckpt"
  evaluator = Evaluator(
    path_weights,
    load_epoch=args.load_epoch,
    vid_path=args.vid_path,
    show=args.show,
    save=args.save,
    deterministic=args.deterministic,
    eval_num=args.eval_num,
    verbose=args.verbose,
    show_predict=args.show_predict,
    fixed_window=args.fixed_window,
    reward_ocr_interval=args.reward_ocr_interval,
    time_ocr_interval=args.time_ocr_interval,
    elixir_ocr_interval=args.elixir_ocr_interval,
    save_video=args.save_video,
    save_video_interval=args.save_video_interval,
    save_frame_log=args.save_frame_log,
  )
  evaluator.eval()
