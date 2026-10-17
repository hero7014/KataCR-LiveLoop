""" Test model on video data """
import cv2, time
import katacr.build_dataset.constant as const
from pathlib import Path
from katacr.utils import colorstr
from katacr.policy.perceptron.sar_builder import SARBuilder
from katacr.yolov8.predict import ImageAndVideoLoader, Stopwatch, second2str

class VideoEnv:
  def __init__(
      self, vid_path, video_interval=3, action_freq=2, show=True, save=False,
      verbose=True, reward_ocr_interval=10,
      time_ocr_interval=3, elixir_ocr_interval=2,
      path_save_dir=None, save_video=False, save_video_interval=10
    ):
    self.show, self.save = show, save
    self.action_freq, self.verbose = action_freq, verbose
    self.save_video = save_video
    self.save_video_interval = max(1, int(save_video_interval))
    self.path_save_dir = Path(path_save_dir) if path_save_dir is not None else None
    self.vid_writer = None
    self.path_save_vid = None
    self.dt = {
      'visual_fusion': None,
      'action_update': None,
      'state_update': None,
      'reward_update': None,
      'sar_update': None,
      'sar_get': None,
      'sar_total': None,
    }
    self.done = True
    self.ds = ImageAndVideoLoader(vid_path, video_interval=video_interval, cvt_part2=False)
    self.sar_builder = SARBuilder(
      reward_ocr_interval=reward_ocr_interval,
      time_ocr_interval=time_ocr_interval,
      elixir_ocr_interval=elixir_ocr_interval)
    self.idx2card = self.sar_builder.visual_fusion.classifier.idx2card
    self.open_window = False
    self.reset()
  
  def _show_dt(self):
    if not self.verbose: return
    print(colorstr("Time used (Env):"), end='')
    tot = 0
    for k, t in self.dt.items():
      if t is None: continue
      print(f" {k}={t*1e3:.1f}ms", end='')
      tot += t
    print()
  
  def get_sar(self):
    for p, x, cap, info in self.ds:
      results = self.sar_builder.update(x)
      if results is None: continue
      dt = results[1]
      for k, t in zip(['visual_fusion', 'action_update', 'state_update', 'reward_update'], dt):
        self.dt[k] = t
      for old_key in list(self.dt):
        if old_key.startswith('visual.'):
          self.dt.pop(old_key)
      for k, t in getattr(self.sar_builder, 'last_visual_fusion_detail', {}).items():
        self.dt[f'visual.{k}'] = t
      for k, t in getattr(self.sar_builder, 'last_state_update_detail', {}).items():
        self.dt[f'state.{k}'] = t
      self.dt['sar_update'] = sum(dt)
      self.count += 1
      should_write_video = self.save_video and self.count % self.save_video_interval == 0
      if self.show or should_write_video:
        rimg = self.sar_builder.render()
        rimg_size = rimg.shape[:2][::-1]
        if self.show:
          if not self.open_window:
            self.open_window = True
            cv2.namedWindow('Detection', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow('Detection', rimg_size)
          cv2.imshow('Detection', rimg)
          cv2.waitKey(1)
        if should_write_video:
          self._write_render_video(rimg, cap)
      if self.count % self.action_freq == 0:
        s, a, r, dt = self.sar_builder.get_sar(verbose=False)
        self.dt['sar_get'] = dt
        for old_key in list(self.dt):
          if old_key.startswith('sar.'):
            self.dt.pop(old_key)
        for k, t in getattr(self.sar_builder, 'last_get_sar_detail', {}).items():
          if isinstance(t, dict):
            for kk, vv in t.items():
              self.dt[f'sar.{k}.{kk}'] = vv
          else:
            self.dt[f'sar.{k}'] = t
        self.dt['sar_total'] = self.dt['sar_get'] + self.dt['sar_update']
        done = self.ds.frame == self.ds.total_frame // 2 * 2
        if done and self.ds.total_frame % 2 == 1:
          next(self.ds)  # skip last frame
        # print(f"Frame: {self.ds.frame=} {self.ds.total_frame=}", info)
        if a['card_id']:
          print(colorstr('red', 'bold', f"Target Action (time={s['time']},frame={self.count}):"), a)
        return s, a, r, done

  def _write_render_video(self, rimg, cap):
    if self.path_save_dir is None:
      return
    if self.vid_writer is None:
      self.path_save_dir.mkdir(parents=True, exist_ok=True)
      self.path_save_vid = self.path_save_dir / f"episode_{getattr(self, 'episode', 1)}_detection.mp4"
      fps = 10
      if cap:
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        if raw_fps and raw_fps > 0:
          fps = max(1, raw_fps / max(1, self.ds.video_interval))
      self.vid_writer = cv2.VideoWriter(
        str(self.path_save_vid), cv2.VideoWriter_fourcc(*'mp4v'),
        fps, rimg.shape[:2][::-1])
      if not self.vid_writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {self.path_save_vid}")
    self.vid_writer.write(rimg)

  def close(self):
    if self.vid_writer is not None:
      self.vid_writer.release()
      self.vid_writer = None
  
  def reset(self, auto=False):
    self.close()
    self.count = 0
    self.total_reward = 0
    self.episode = getattr(self, 'episode', 0) + 1
    self.sar_builder.reset()
    s, a, r, done = self.get_sar()
    self.total_reward += r
    self._show_dt()
    return s, a, r
  
  def step(self, action, max_delay=5, prob_img=None):
    """
    Args:
      action (np.ndarray): [select_index, position_xy, (delay)], shape=(3,) or (4,)
      max_delay (int): The max delay time.
    """
    assert self.done, "self.done=True, need reset first!"
    s, a, r, done = self.get_sar()
    self.total_reward += r
    if action[0] and (len(action) == 3 or (len(action) == 4 and action[-1] <= max_delay)):
      print(colorstr('yellow', 'bold', f"Predict Action (time={s['time']},frame={self.count},delay={action[-1]}):"), action)
    self._show_dt()
    return s, a, r, done, {
      'total_reward': self.total_reward,
      'visual_count': self.count,
      'source_frame': getattr(self.ds, 'frame', None),
      'total_frame': getattr(self.ds, 'total_frame', None),
      'timings': dict(self.dt),
    }

if __name__ == '__main__':
  vid_path = "/home/yy/Videos/CR_Videos/test/golem_ai/3.mp4"
  env = VideoEnv(vid_path, save=False, verbose=False)
  env.reset()
  while True:
    # i = int(input())
    s, a, r, done = env.step([0, 9, 22])
    if done:
      env.reset()
