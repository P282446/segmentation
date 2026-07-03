import torch as torch
from torchvision.transforms.v2 import Normalize
from skimage import io, exposure
import numpy as np
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageSegmentation
from media.image_processing import duoImage, open_file
from matplotlib import pyplot as plt
import cv2
import skimage as ski
from configuration_loader import config
from tqdm import tqdm
import torch.multiprocessing as mp

model = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])


def quantize(array, a, b, threshold):
    array[array >= threshold] = b
    array[array < threshold] = a


# TODO : BATCH THE PREPROCESSING/POSTPROCESSING
def preprocess_image(im, model_input_size):
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    # orig_im_size=im.shape[0:2]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0),
                              size=model_input_size,
                              mode='bilinear')
    image = torch.divide(im_tensor, 255.0)
    image = (image)
    return image.to(device)


def postprocess_image(result, im_size):
    result = torch.squeeze(
        F.interpolate(result, size=im_size, mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2,
                                      0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array

def find_body_pixels_batch(images, landmarks_list, threshold=128):
    global model
    if model is None:
        model = AutoModelForImageSegmentation.from_pretrained(
            "models/rmbg1.4.model/", trust_remote_code=True)

        model.to(device)
    
    processed_images = []
    batch = []
    batch_orig_im_sizes = []
    
    for idx, im in enumerate(images):
        orig_im_size = im.size[::-1]
        batch.extend(preprocess_image(im.opencv_version, orig_im_size))
        batch_orig_im_sizes.append(orig_im_size)
        
   
    batch = torch.stack(batch)
    with torch.no_grad():
        result = model(batch)

    for idx, (im, orig_im_size) in enumerate(zip(images, batch_orig_im_sizes)):
        result_image = postprocess_image (torch.unsqueeze(result[0][idx][0] , 0), batch_orig_im_sizes[idx])
        result_image[np.where(result_image < threshold)] = 0
        result_image[np.where(result_image >= threshold)] = 255
        
        if len(landmarks_list) > idx and len(landmarks_list[idx]) > 0:
            landmarks = landmarks_list[idx]
            out = [
                364, 400, 358, 457, 129, 170, 135, 183, 216, 78, 203, 20, 101, 66, 32,
                179, 100, 70, 102, 88, 89, 67, 90, 315, 319, 291, 318, 317, 331, 295,
                409, 335, 374, 233, 340, 305, 446, 413
            ]
            outer_bounds = [landmarks[o] for o in out]
            top = min([outer_bound[1] for outer_bound in outer_bounds])
            left = min([outer_bound[0] for outer_bound in outer_bounds])
            bottom = max([outer_bound[1] for outer_bound in outer_bounds])
            right = max([outer_bound[0] for outer_bound in outer_bounds])
            center = int((top + bottom) // 2), int((left + right) // 2)
            processed_images.append(
                duoImage(Image.fromarray(
                    quick_find_connected_component(result_image, center)))
            )
        else:
            processed_images.append(duoImage(Image.fromarray(result_image)))
    return processed_images




def sliding_window_generator(generator, window_size):
    window = []
    for item in generator:
        window.append(item)
        if len(window) == window_size:
            yield window[:]
            window.pop(0)
    while len(window) > 0:
        yield window[:]
        window.pop(0)

def generate_consensus_image(window):
    window_t = [torch.from_numpy(img).to(device) for img in window]
    stack = torch.stack(window_t, dim=0)
    counts = torch.sum(stack == 255, dim=0)
    consensus_mask = torch.zeros_like(counts, device=device)
    consensus_mask[counts > len(window) // 2] = 255
    current = window_t[len(window) // 2]
    d = 1
    neighborhood_sum = torch.zeros_like(current, device=device)
    for i in range(-d, d + 1):
        for j in range(-d, d + 1):
            neighborhood_sum = neighborhood_sum + torch.roll(consensus_mask, (i, j), dims=(0, 1))
    consensus_mask[neighborhood_sum > 0] = 255
    return consensus_mask


def quick_find_connected_component(array, point, target_size=(100, 100)):
    width, height = array.shape[1], array.shape[0]
    new_w, new_h = width, height
    new_w = width//3
    new_h = height//3
    new_w = int(new_w)
    new_h = int(new_h)
    w_factor = new_w / width
    h_factor = new_h / height
    array_ = cv2.resize(array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    quantize(array_, 0, 255, 128)
    labeled_image = ski.measure.label(array_, connectivity=1, return_num=False)
    point_label = labeled_image[int(point[0] * h_factor),
                                int(point[1] * w_factor)]
    component = np.zeros_like(array_, dtype=np.uint8)
    component[labeled_image == point_label] = 255
    scaled_connected_component = np.zeros((height, width))

    component = cv2.resize(component, (width, height),
                           interpolation=cv2.INTER_LINEAR)
    mask = (array > 0) & (component > 0)

    scaled_connected_component[mask] = 255
    return scaled_connected_component


def omniscient_filter_body_pixels(generate_mask, num_masks, inertia=4):

    window_generator = sliding_window_generator(generate_mask, 2 * inertia + 1)
    i = 0
    while i < num_masks - inertia:
        window = next(window_generator)
        if i == 0:
            for j in range(inertia):
                window_ = window[0:j + inertia + 1]
                yield np.array(generate_consensus_image(window_).cpu())
        yield np.array(generate_consensus_image(window).cpu())
        i += 1


def worker(window):
    return np.array(generate_consensus_image(window).cpu())

def omniscient_filter_body_pixels_parallelized (generate_mask, num_masks, inertia=4, num_processes=4):
    window_generator = sliding_window_generator(generate_mask, 2 * inertia + 1)
    pool = mp.Pool(num_processes)
    i = 0
    results = []
    for i in tqdm(range (num_masks - inertia)):
        window = next(window_generator)
        if i == 0:
            for j in range(inertia):
                window_ = window[0:j + inertia + 1]
                results.append(pool.apply_async(worker, args=(window_,)))
        results.append(pool.apply_async(worker, args=(window,)))
        i += 1



    for res in tqdm(results):
        res.wait()
        yield res.get()
