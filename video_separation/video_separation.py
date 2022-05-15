import gc
import cv2
import numpy as np

class VideoSeparation:
  def __init__(self, path_video, window_size=30, min_val=2.5, smooth_amount=100):
    self.window_size = window_size
    self.min_val = min_val
    self.smooth_amount = smooth_amount
    self.path_to_video = path_video
    video_path = self.path_to_video.split('.')[-1]
    self.path_to_stabilized_video = video_path + "_stabilized.avi"
    self.stabilizeVideo(self.smooth_amount, self.path_to_video, self.path_to_stabilized_video)
    self.stabilized_video = cv2.VideoCapture(self.path_to_stabilized_video)
    self.path_to_transmission_video = video_path + "_transmission.avi"
    self.transmission_video = self.get_transmission()
    self.path_to_reflection_video = video_path + "_reflection.avi"
    self.reflection_video = self.get_reflection()
    
  def get_transmission(self):
    self.minimum_layer = self.process_video_separation(self.stabilized_video, self.extract_minimum_layer, self.path_to_transmission_video)
    self.transmission_video = cv2.VideoCapture(self.path_to_transmission_video)
    return self.transmission_video


  def get_reflection(self):
    self.maximum_layer = self.extract_maximum_layer(self.stabilized_video, self.minimum_layer, self.path_to_reflection_video)
    self.reflection_video = cv2.VideoCapture(self.path_to_reflection_video)
    return self.reflection_video

  def save_video(video, fps, path):
    h, w, _ = video[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))

    for i in range(len(video)):
      out.write(video[i])

    out.release()
    cv2.destroyAllWindows()

  def process_video_separation(video, extraction_function, path):
    separated_video = dict()
    window = list()
    ret = True
    separated_frame_index = 0
    fps = round(video.get(cv2.CAP_PROP_FPS))
    for i in range(fps):
      ret, frame = video.read()
      if ret:
        window.append(frame)

      else:
        return None

    extraction_function(separated_frame_index, separated_video, window)
    separated_frame_index += 1
    i = fps
    while ret:
      ret, frame = video.read()
      window = window[1:]
      if frame is not None:
        window.append(frame)
        extraction_function(separated_frame_index, separated_video, window)
        separated_frame_index += 1

      i += 1

    separated_video = np.array(list(dict(sorted(separated_video.items())).values()))
    self.save_video(separated_video, fps, path)
    return separated_video

  def extract_minimum_layer(frame_number, separated_video, window):
    separated_image = np.min(window, axis=0).astype(np.uint8)
    separated_video[frame_number] = separated_image

  def extract_maximum_layer(video, min_layer, save_path):
    fps = round(video.get(cv2.CAP_PROP_FPS))
    origin_video = list()
    ret, frame = video.read()
    i = 0
    while ret and i < min_layer.shape[0]:
      origin_video.append(frame)
      ret, frame = video.read()
      i+=1

    # origin_video = np.rot90(origin_video, k=-1, axes=(1,2))
    origin_video = np.array(origin_video)
    reflection_video = origin_video - min_layer
    self.save_video(reflection_video, fps, save_path)
    return reflection_video
    
  def movingAverage(curve, radius): 
    window_size = 2 * radius + 1
    # Define the filter 
    f = np.ones(window_size)/window_size 
    # Add padding to the boundaries 
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge') 
    # Apply convolution 
    curve_smoothed = np.convolve(curve_pad, f, mode='same') 
    # Remove padding 
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed 

  def smooth(trajectory, smooth_amount): 
    smoothed_trajectory = np.copy(trajectory) 
    # Filter the x, y and angle curves
    for i in range(3):
      smoothed_trajectory[:,i] = self.movingAverage(trajectory[:,i], radius=smooth_amount)

    return smoothed_trajectory

  def fixBorder(frame):
    s = frame.shape
    # Scale the image 8% without moving the center
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.08)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

  def stabilizeVideo(smooth_amount, path_to_video, path_to_stabilized_video):
    # smooth_amount - The larger the more stable the video, but less reactive to sudden panning

    # Read input video
    cap = cv2.VideoCapture(path_to_video) 

    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get frames per second (fps)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec for output video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    # Set up output video
    out = cv2.VideoWriter(path_to_stabilized_video, fourcc, fps, (h,w))

    # Read first frame
    _, prev = cap.read() 

    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) 

    # Pre-define transformation-store array
    transforms = np.zeros((n_frames-1, 3), np.float32) 

    for i in range(n_frames-2):
      # Detect feature points in previous frame
      prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                        maxCorners=200,
                                        qualityLevel=0.01,
                                        minDistance=30,
                                        blockSize=3)

      # Read next frame
      success, curr = cap.read() 
      if not success: 
        break 

      # Convert to grayscale
      curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 

      # Calculate optical flow (i.e. track feature points)
      curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None) 

      # Sanity check
      assert prev_pts.shape == curr_pts.shape 

      # Filter only valid points
      idx = np.where(status==1)[0]
      prev_pts = prev_pts[idx]
      curr_pts = curr_pts[idx]

      #Find transformation matrix
      # m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) #will only work with OpenCV-3 or less
      m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts) 

      # Extract traslation
      dx = m[0,2]
      dy = m[1,2]

      # Extract rotation angle
      da = np.arctan2(m[1,0], m[0,0])

      # Store transformation
      transforms[i] = [dx,dy,da]

      # Move to next frame
      prev_gray = curr_gray

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0) 

    # Create variable to store smoothed trajectory
    smoothed_trajectory = self.smooth(trajectory, smooth_amount) 

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    # Reset stream to first frame 
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 

    # Write n_frames-1 transformed frames
    for i in range(n_frames-2):
      # Read next frame
      success, frame = cap.read() 
      if not success:
        break

      # Extract transformations from the new transformation array
      dx = transforms_smooth[i,0]
      dy = transforms_smooth[i,1]
      da = transforms_smooth[i,2]

      # Reconstruct transformation matrix accordingly to new values
      m = np.zeros((2,3), np.float32)
      m[0,0] = np.cos(da)
      m[0,1] = -np.sin(da)
      m[1,0] = np.sin(da)
      m[1,1] = np.cos(da)
      m[0,2] = dx
      m[1,2] = dy

      # Apply affine wrapping to the given frame
      frame_stabilized = cv2.warpAffine(frame, m, (w,h))

      # Fix border artifacts
      frame_stabilized = self.fixBorder(frame_stabilized) 

      cv2.waitKey(10)
      out.write(cv2.rotate(frame_stabilized, cv2.ROTATE_90_CLOCKWISE))

    # Release video
    cap.release()
    out.release()
    # Close windows
    cv2.destroyAllWindows()
