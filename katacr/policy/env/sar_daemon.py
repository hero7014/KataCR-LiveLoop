import numpy as np
from pathlib import Path
import multiprocessing, cv2, time
import os
from katacr.build_dataset.utils.split_part import process_part
from katacr.policy.env.utils import tap_screen
from katacr.utils.ffmpeg.format_conversion import compress_video
import shutil
DESTROY_FRAME_DELTA_THRE = 10  # wait 10*0.3s after game terminal

path_root = Path(__file__).parents[3]
path_save_dir = path_root / "logs/interaction" / time.strftime("%Y%m%d_%H:%M:%S")

class SARDaemon:
  WAIT_FOR_NEXT_EPISODE = 1
  AUTO_START_NEXT_EPISODE = 2
  def __init__(
      self,
      q_reset: multiprocessing.Queue,
      q_sar: multiprocessing.Queue,
      q_info: multiprocessing.Queue,
      q_prob_img: multiprocessing.Queue,
      show: bool = True, save: bool = False, interval: int = 2
    ):
    from katacr.policy.perceptron.sar_builder import SARBuilder
    self.q_reset, self.q_sar, self.q_info, self.q_prob_img = q_reset, q_sar, q_info, q_prob_img
    self.show, self.save = show, save
    self.interval = interval
    self.sar_builder = SARBuilder()
    self.q_info.put({'idx2card': self.sar_builder.visual_fusion.classifier.idx2card, 'path_save_dir': path_save_dir})
    self.ocr = self.sar_builder.action_builder.ocr  # Share OCR with action text OCR
    self.source = os.getenv("KATACR_INTERACT_SOURCE", "camera")
    self.tap_offset = (0, 0)
    self.cap = None
    if self.source == "adb":
      from scripts.adb_probe import normalize_portrait_frame, screencap
      self.adb_normalize = normalize_portrait_frame
      self.adb_screencap = screencap
      self.adb = os.getenv("KATACR_ADB", "adb")
      self.adb_serial = os.getenv("KATACR_ADB_SERIAL", "127.0.0.1:7555")
      self.target_ratio = float(os.getenv("KATACR_ADB_TARGET_RATIO", "2.16"))
      self.adb_normalize_mode = os.getenv("KATACR_ADB_NORMALIZE", "resize_width")
    else:
      self.cap = cv2.VideoCapture(2)
      assert self.cap.isOpened(), "The phone stream can't connect!"
    self.open_window = False
    self.vid_writer = None
    if self.save:
      path_save_dir.mkdir(exist_ok=True, parents=True)
    self.episode = 0
    self.terminal = True
    self.first_terminal_time = None
    self.total_reward = 0
    self.run()
  
  def _read_img(self):
    self.timestamp = time.time()
    if self.source == "adb":
      raw, _, _ = self.adb_screencap(self.adb, serial=self.adb_serial)
      self.raw_img = raw
      if self.adb_normalize_mode == "resize_width":
        h, w = raw.shape[:2]
        target_w = int(h / self.target_ratio)
        img = cv2.resize(raw, (target_w, h), interpolation=cv2.INTER_AREA)
        self.tap_offset = (0, 0)
        self.tap_scale = (w / target_w, 1)
      else:
        img, norm_info = self.adb_normalize(raw, self.target_ratio)
        self.tap_offset = (norm_info.get("crop_x0", 0), norm_info.get("crop_y0", 0))
        self.tap_scale = (1, 1)
      return img
    _, img = self.cap.read()
    self.tap_offset = (0, 0)
    self.tap_scale = (1, 1)
    return img

  def _is_open(self):
    return True if self.source == "adb" else self.cap.isOpened()

  def _wait_next_episode(self):
    """ Wait for entering one episode """
    if self.source == "adb" and os.getenv("KATACR_MUMU_SIMPLE_START", "0") == "1":
      print("MuMu simple start: taking current frame as episode start.")
      return self._read_img()
    episode_start_time = time.time()
    last_wait_log_time = episode_start_time
    while self._is_open():
      img = self._read_img()
      t = self.ocr.process_part1(process_part(img, 1))
      ep_flag = self.ocr.process_center_texts(img)
      if (
        self.source == "adb"
        and os.getenv("KATACR_MUMU_SIMPLE_START", "0") == "1"
        and np.isfinite(t)
        and 0 < t <= 180
      ):
        return img
      if ep_flag == self.ocr.START_EPISODE_FLAG and t < 10:
        return img
      if self.source == "adb" and os.getenv("KATACR_MUMU_SIMPLE_START", "0") == "1":
        try:
          from scripts.adb_ui_flow import detect_page_state
          ui_state = detect_page_state(getattr(self, "raw_img", img)).value
        except Exception:
          ui_state = None
        if time.time() - episode_start_time > 4 and ui_state not in ("home", "post_match"):
          return img
      time.sleep(0.05)
      if time.time() - last_wait_log_time > 2:
        print("Wait for opening new episode...")
        last_wait_log_time = time.time()

  def _check_terminal(self):
    if self.source == "adb" and os.getenv("KATACR_MUMU_SIMPLE_START", "0") == "1":
      battle_time = getattr(self.sar_builder.state_builder, "time", None)
      try:
        from scripts.adb_ui_flow import detect_page_state
        ui_state = detect_page_state(getattr(self, "raw_img", self.img)).value
        min_terminal_time = int(os.getenv("KATACR_MUMU_MIN_TERMINAL_TIME", "20"))
        if ui_state == "post_match" and battle_time is not None and battle_time >= min_terminal_time:
          self.terminal = True
          while not self.q_sar.empty(): self.q_sar.get()
          return
      except Exception:
        pass
      if battle_time is not None and battle_time > int(os.getenv("KATACR_MUMU_MAX_BATTLE_TIME", "190")):
        self.terminal = True
        while not self.q_sar.empty(): self.q_sar.get()
      return
    if self.first_terminal_time is None:
      ep_flag = self.ocr.process_center_texts(self.img)
      if ep_flag == self.ocr.END_EPISODE_FLAG:
        self.first_terminal_time = time.time()
        while not self.q_sar.empty(): self.q_sar.get()
    elif time.time() - self.first_terminal_time > DESTROY_FRAME_DELTA_THRE * 0.3:
        self.first_terminal_time = None
        self.terminal = True

  def _start_new_episode(self):
    if self.source == "adb" and os.getenv("KATACR_MUMU_SIMPLE_START", "0") == "1":
      img = self._read_img()
      t = self.ocr.process_part1(process_part(img, 1))
      if np.isfinite(t) and 0 < t <= 180:
        return
      # Current MuMu lobby layout: tap the visible battle button and wait for the match to load.
      tap_screen((360, 1000), relative=False, delay=3.0)
      return
    # img_size = self.img.shape[:2][::-1]  # (1080, 2400)
    time.sleep(10)  # wait for OK button
    img_size = (1080, 2400)
    # End episode OK button
    tap_screen((534/1080,1940/2400), img_size=img_size, delay=10.0)
    # Tap empty space
    tap_screen((534/1080,65/2400), img_size=img_size, delay=0.5)
    # Menu button
    tap_screen((980/1080,320/2400), img_size=img_size, delay=0.5)
    # Training camp button
    tap_screen((648/1080,746/2400), img_size=img_size, delay=0.5)
    # Start episode OK button
    tap_screen((730/1080,1400/2400), img_size=img_size)

  def run(self):
    while True:
      ### Check If Reset ###
      reset = self.terminal
      while not self.q_reset.empty() or self.terminal:
        if self.vid_writer is not None:
          self.vid_writer.release()
          # self.vid_writer_org.release()
          self.vid_writer = None
          # self.vid_writer_org = None
          # print("RELEASE mp4 videos!!!")
          compress_video(self.path_save_vid)
        if self.terminal and self.q_reset.empty():
          time.sleep(0.05)
          continue
        self.terminal = False
        self.total_reward = 0
        reset = True
        reset_id = self.q_reset.get()
        if reset_id == self.AUTO_START_NEXT_EPISODE:
          # print("Start new episode AUTO!")
          self._start_new_episode()
      if reset:
        self.episode += 1
        self.count = 0
        self.sar_builder.reset()
        self.img = self._wait_next_episode()
      else:
        self.img = self._read_img()
      ### Get SAR and Terminal ###
      while self.q_sar.qsize() > 1: self.q_sar.get()
      results = self.sar_builder.update(self.img)
      if results is None: continue
      dt = results[1]
      self.count += 1
      if self.count % self.interval == 0:
        info = {'dt': {}, 'timestamp': self.timestamp, 'img_size': self.img.shape[:2][::-1], 'tap_offset': self.tap_offset, 'tap_scale': self.tap_scale}
        info['raw_cards'] = self.sar_builder.last_raw_cards
        info['dt']['sar_update'] = sum(dt)
        s, a, r, dt = self.sar_builder.get_sar(verbose=False)
        self.total_reward += r
        info['dt']['sar_get'] = dt
        info['dt']['sar_total'] = info['dt']['sar_get'] + info['dt']['sar_update']
        info['total_reward'] = self.total_reward
        self._check_terminal()
        self.q_sar.put((s, a, r, self.terminal, info))
      ### Check If Show or Save ###
      if self.show or self.save:
        rimg = self.sar_builder.render()
        rimg_size = rimg.shape[:2][::-1]
        org_img_size = (int(rimg_size[1]/1280*600), rimg_size[1])
        org_img = cv2.resize(self.img, org_img_size)
        while not self.q_prob_img.empty():
          self.prob_img = self.q_prob_img.get()
        if not hasattr(self, 'prob_img'):
          self.prob_img = np.zeros((896, 576, 3), np.uint8)
        # print(rimg.shape, org_img.shape)
        rimg = np.concatenate([org_img, rimg, self.prob_img], 1)
        rimg_size = rimg.shape[:2][::-1]
        if self.show:
          if not self.open_window:
            self.open_window = True
            cv2.namedWindow('Agent', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow('Agent', rimg_size)
            # cv2.namedWindow('Origin', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            # cv2.resizeWindow('Origin', org_img_size)
          cv2.imshow('Agent', rimg)
          # cv2.imshow('Origin', org_img)
          cv2.waitKey(1)
        if self.save:
          if self.vid_writer is None:
            self.path_save_vid = path_save_dir / f"{self.episode}.mp4"
            self.vid_writer = cv2.VideoWriter(str(self.path_save_vid), cv2.VideoWriter_fourcc(*'mp4v'), 10, rimg_size)
            # path_save_vid = path_save_dir / f"{self.episode}_org.mp4"
            # self.vid_writer_org = cv2.VideoWriter(str(path_save_vid), cv2.VideoWriter_fourcc(*'mp4v'), 10, org_img_size)
          self.vid_writer.write(rimg)
          # self.vid_writer_org.write(org_img)
