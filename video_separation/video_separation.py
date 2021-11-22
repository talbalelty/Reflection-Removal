import gc
import cv2
import numpy as np

class VideoSeparation:
  def __init__(self, path_video, window_size=10, min_val=2.5):
    self.window_size = window_size
    self.min_val = min_val
    self.path_to_video = path_video
    self.transmission_video = None
    self.reflection_video = None
    
  def get_transmission(self):
    self.transmission_video = self.__separate_video(self.__extract_transmission_layer)
    return self.transmission_video


  def get_reflection(self):
    self.reflection_video = self.__separate_video(self.__extract_reflection_layer)
    return self.reflection_video

  def save_video(self, video, path):
    h, w, _ = video[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, self.window_size, (w, h))

    for i in range(len(video)):
      out.write(video[i])

    out.release()
    cv2.destroyAllWindows()
    print(path + "\nVideo made successfully")
    
  def __separate_video(self, extract_layer):
    # transmission
    gc.collect()
    video = cv2.VideoCapture(self.path_to_video)
    separated_video = self.__process_video_separation(video, extract_layer)
    
    return separated_video

  def __process_video_separation(self, video, extraction_function):
    separated_video = dict()
    window = list()
    ret = True
    separated_frame_index = 0
    fps = round(video.get(cv2.CAP_PROP_FPS))
    skip_frames = fps // self.window_size
    for i in range(fps):
      ret, frame = video.read()
      if ret:
        if i % skip_frames == 0:
          window.append(frame)

      else:
        return None

    extraction_function(separated_frame_index, separated_video, window)
    separated_frame_index += 1

    i = fps
    while (ret):
      ret, frame = video.read()
      if i % skip_frames == 0:
        window = window[1:]
        if frame is not None:
          window.append(frame)
          extraction_function(separated_frame_index, separated_video, window)
          separated_frame_index += 1

      i += 1

    separated_video = list(dict(sorted(separated_video.items())).values())
    separated_video = np.rot90(separated_video, k=-1, axes=(1,2))    
    return separated_video

  def __extract_transmission_layer(self, frame_number, separated_video, window):
    separated_image = np.median(window, axis=0).astype(np.uint8)
    separated_video[frame_number] = separated_image

  def __extract_reflection_layer(self, frame_number, separated_video, window):
    grey_window = list()

    for i, frame in enumerate(window):
      grey_window.append(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))

    mean = np.mean(grey_window, axis=0)
    threshold = self.min_val * np.std(grey_window, axis=0)
    mask = (grey_window[0] < mean - threshold) | (mean + threshold < grey_window[0])
    mask = mask[:, :, np.newaxis]
    mask = np.repeat(mask, 3, axis=2)
    separated_video[frame_number] = np.where(mask, window[0], 0)
