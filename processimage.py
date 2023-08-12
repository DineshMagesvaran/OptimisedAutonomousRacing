import numpy as np
from skimage import color

class processimage:
    def process_image(obs):
        obs1 = obs.astype(np.uint8)
        obs_gray = color.rgb2gray(obs1)
        obs_gray[abs(obs_gray - 0.68616) < 0.0001] = 1
        obs_gray[abs(obs_gray - 0.75630) < 0.0001] = 1
        return 2 * obs_gray - 1