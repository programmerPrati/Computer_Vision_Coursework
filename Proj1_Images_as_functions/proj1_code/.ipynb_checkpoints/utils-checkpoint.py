#!/usr/bin/python3

import copy
from typing import Any, List, Tuple
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def PIL_resize(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
  """
    Args:
    - img: Array representing an image
    - size: Tuple representing new desired (width, height)

    Returns:
    - img
  """
  pil = numpy_arr_to_PIL_image(img, scale_to_255=True)
  pil = pil.resize(size, resample=Image.BILINEAR)
  img = PIL_image_to_numpy_arr(pil)
  return img


def PIL_image_to_numpy_arr(img, downscale_by_255=True):
  """
    Args:
    - img
    - downscale_by_255

    Returns:
    - img
  """
  arr = np.asarray(img).astype(np.float32)
  if downscale_by_255:
    arr /= 255.0
  return arr


def vis_image_scales_numpy(image: np.ndarray) -> np.ndarray:
  """
    This function will display an image at different scales (zoom factors). The
    original image will appear at the far left, and then the image will
    iteratively be shrunk by 2x in each image to the right.

    This is a particular effective way to simulate the perspective effect, as
    if viewing an image at different distances. We thus use it to visualize
    hybrid images, which represent a combination of two images, as described
    in the SIGGRAPH 2006 paper "Hybrid Images" by Oliva, Torralba, Schyns.

    Args:
    - image: Array of shape (H, W, C)

    Returns:
    - img_scales: Array of shape (M, K, C) representing horizontally stacked
      images, growing smaller from left to right.
      K = W + int(1/2 W + 1/4 W + 1/8 W + 1/16 W) + (5 * 4)
  """
  # Normalize to 3D for stacking logic
  was_gray = False
  if image.ndim == 2:
    image = image[:, :, None]
    was_gray = True

  original_height = image.shape[0]
  original_width = image.shape[1]
  num_colors = image.shape[2]
  img_scales = np.copy(image)
  cur_image = np.copy(image)

  scales = 5
  scale_factor = 0.5
  padding = 5

  new_h = original_height
  new_w = original_width

  for _ in range(2, scales+1):
    # add padding
    img_scales = np.hstack((
      img_scales,
      np.ones((original_height, padding, num_colors), dtype=np.float32))
    )

    new_h = max(1, int(round(scale_factor*new_h)))
    new_w = max(1, int(round(scale_factor*new_w)))
    # downsample image iteratively
    cur_image = PIL_resize(cur_image, size=(new_w, new_h))
    if cur_image.ndim == 2:
      cur_image = cur_image[:, :, None]

    # pad the top to append to the output
    h_pad = original_height - cur_image.shape[0]
    if h_pad < 0:
      cur_image = cur_image[-h_pad:, ...]
      h_pad = 0
    pad = np.ones((h_pad, cur_image.shape[1], num_colors), dtype=np.float32)
    tmp = np.vstack((pad, cur_image))
    img_scales = np.hstack((img_scales, tmp))

  if was_gray:
    img_scales = img_scales[:, :, 0]
  return img_scales


def im2single(im: np.ndarray) -> np.ndarray:
  """
    Args:
    - img: uint8 array of shape (m,n,c) or (m,n) and in range [0,255]

    Returns:
    - im: float or double array of identical shape and in range [0,1]
  """
  im = im.astype(np.float32) / 255.0
  return im

def single2im(im: np.ndarray) -> np.ndarray:
  """
    Args:
    - im: float or double array of shape (m,n,c) or (m,n) and in range [0,1]

    Returns:
    - im: uint8 array of identical shape and in range [0,255]
  """
  arr = (im * 255.0).round().clip(0, 255).astype(np.uint8)
  return arr


def numpy_arr_to_PIL_image(img: np.ndarray, scale_to_255: bool = False) -> Image.Image:
  """
    Args:
    - img: in [0,1] or [0,255]

    Returns:
    - PIL Image
  """
  arr = img.copy()
  if scale_to_255:
    arr = (arr * 255.0).round().clip(0, 255)
  arr = np.uint8(arr)

  if arr.ndim == 2:
    return Image.fromarray(arr, mode="L")
  if arr.ndim == 3 and arr.shape[2] == 3:
    return Image.fromarray(arr, mode="RGB")
  if arr.ndim == 3 and arr.shape[2] == 4:
    return Image.fromarray(arr, mode="RGBA")
  return Image.fromarray(arr)


def load_image(path: str) -> np.ndarray:
  """
    Args:
    - path: string representing a file path to an image

    Returns:
    - float or double array of shape (m,n,c) or (m,n) and in range [0,1],
      representing an RGB image
  """
  p = Path(path)
  with Image.open(p) as im:
    if im.mode in ("RGBA", "P"):
      im = im.convert("RGB")
    elif im.mode not in ("L", "RGB"):
      im = im.convert("RGB")
    img = np.asarray(im)
  float_img_rgb = im2single(img)
  return float_img_rgb


def save_image(path: str, im: np.ndarray) -> bool:
  """
    Args:
    - path: string representing a file path to an image
    - img: numpy array

    Returns:
    - retval indicating write success
  """
  img = copy.deepcopy(im)
  if img.dtype != np.uint8:
    img = single2im(img)
  pil_img = numpy_arr_to_PIL_image(img, scale_to_255=False)
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  pil_img.save(path)
  return True


def write_objects_to_file(fpath: str, obj_list: List[Any]):
  """
    If the list contents are float or int, convert them to strings.
    Separate with carriage return.

    Args:
    - fpath: string representing path to a file
    - obj_list: List of strings, floats, or integers to be written out to a file, one per line.

    Returns:
    - None
  """
  obj_list = [str(obj) + '\n' for obj in obj_list]
  with open(fpath, 'w', encoding='utf-8') as f:
    f.writelines(obj_list)