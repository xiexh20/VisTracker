"""
basic operations for the dataset

Author: Xianghui Xie
Date: April 02, 2023
Cite: Visibility Aware Human-Object Interaction Tracking from Single RGB Camera. CVPR'2023
"""
import os
import numpy as np
import cv2, json
cv2.setNumThreads(1)
import os.path as osp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from psbody.mesh import Mesh
import torchvision.transforms as transforms
from PIL.ImageFilter import GaussianBlur


# SCRATCH_PATH = "/scratch/inf0/user/xxie/behave/" # SSD file system, used to load RGB images
# SCRATCH_PATH = "/BS/xxie-4/static00/behave-fps30/" # for rebutal, use old data path
# OLD_PATH = "/BS/xxie-4/static00/behave-fps30/" # old 30fps data

class BaseDataset(Dataset):
    def __init__(self, data_paths, batch_size,
                 num_workers, dtype=np.float32,
                 aug_blur=0.0):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_paths = data_paths
        self.dtype = dtype

        # data augmentation
        self.aug_blur = aug_blur

    def init_others(self):
        pass

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # ret = self.get_item(idx)
        # return ret
        try:
            ret = self.get_item(idx)
            return ret
        except Exception as e:
            print(e)
            ridx = np.random.randint(0, len(self.data_paths))
            print(f"failed on {self.data_paths[idx]}, retrying {self.data_paths[ridx]}")
            return self[ridx]

    def get_item(self, idx):
        raise NotImplemented

    def load_j2d(self, rgb_file):
        """
        load 2D body keypoints, in original image coordinate (before any crop or scale)
        :param rgb_file:
        :return:
        """
        json_path = rgb_file.replace('.color.jpg', '.color.json')
        data = json.load(open(json_path))
        J2d = np.array(data["body_joints"]).reshape((-1, 3))
        return J2d

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)

    def get_loader(self, shuffle=True, rank=-1, world_size=-1):
        if world_size>0:
            "loader for multiple gpu training"
            sampler = DistributedSampler(
                    self,
                    num_replicas=world_size,
                    rank=rank)
            loader = DataLoader(dataset=self, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers,
                                sampler=sampler,
                                pin_memory=True,
                                drop_last=True)
            return loader

        else:
            return DataLoader(
                self, batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=shuffle,
                worker_init_fn=self.worker_init_fn,
                drop_last=False)

    def load_masks(self, rgb_file, flip=False):
        person_mask_file = rgb_file.replace('.color.jpg', ".person_mask.png")
        if not osp.isfile(person_mask_file):
            person_mask_file = rgb_file.replace('.color.jpg', ".person_mask.jpg")
        obj_mask_file = None
        for pat in [".obj_rend_mask.png", ".obj_rend_mask.jpg", ".obj_mask.png", ".obj_mask.jpg"]:
            obj_mask_file = rgb_file.replace('.color.jpg', pat)
            if osp.isfile(obj_mask_file):
                break

        # if OLD_PATH in rgb_file:
        #     # print("Warning: loading from old object masks!")
        #     # import datetime
        #     # assert '2022-09-27' in str(datetime.datetime.now()), 'please check mask loading!'
        #     # assert old_path in rgb_file
        #     person_mask_file = person_mask_file.replace(OLD_PATH, SCRATCH_PATH)
        #     obj_mask_file = obj_mask_file.replace(OLD_PATH, SCRATCH_PATH)

        # obj_mask_file = rgb_file.replace('.color.jpg', ".obj_rend_mask.jpg")
        # if not osp.isfile(obj_mask_file):
        #     obj_mask_file = rgb_file.replace('.color.jpg', ".obj_mask.jpg")
        #     if not osp.isfile(obj_mask_file):
        #         obj_mask_file = rgb_file.replace('.color.jpg', ".obj_mask.png")
        # person_mask = cv2.imread(person_mask_file, cv2.IMREAD_GRAYSCALE) # slower???
        # obj_mask = cv2.imread(obj_mask_file, cv2.IMREAD_GRAYSCALE)
        person_mask = np.array(Image.open(person_mask_file))
        obj_mask = np.array(Image.open(obj_mask_file))
        # assert person_mask.ndim == 2
        # assert obj_mask.ndim == 2

        assert not flip
        if flip:
            person_mask = self.flip_image(person_mask)
            obj_mask = self.flip_image(obj_mask)

        return person_mask, obj_mask

    def flip_image(self, img):
        img = Image.fromarray(img)
        flipped = transforms.RandomHorizontalFlip(p=1.0)(img)
        img = np.array(flipped)
        return img

    def masks2bbox(self, masks, thres=127):
        """
        convert a list of masks to an bbox of format xyxy
        :param masks:
        :param thres:
        :return:
        """
        mask_comb = np.zeros_like(masks[0])
        for m in masks:
            mask_comb += m
        mask_comb = np.clip(mask_comb, 0, 255)
        ret, threshed_img = cv2.threshold(mask_comb, thres, 255, cv2.THRESH_BINARY)
        contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bmin, bmax = np.array([50000, 50000]), np.array([-100, -100])
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            bmin = np.minimum(bmin, np.array([x, y]))
            bmax = np.maximum(bmax, np.array([x+w, y+h]))
        return bmin, bmax

    def center_from_masks(self, obj_mask, person_mask, rgb_file, ret_width=False):
        """compute the cropping center from human and object masks"""
        bmin, bmax = self.masks2bbox([person_mask, obj_mask])  # crop using full bbox
        crop_center = (bmin + bmax) // 2
        assert np.sum(crop_center > 0) == 2, 'invalid bbox found'
        ih, iw = person_mask.shape[:2]
        assert crop_center[0] < iw and crop_center[0] > 0, 'invalid crop center value {} for image {}'.format(
            crop_center, rgb_file)
        assert crop_center[1] < iw and crop_center[1] > 0, 'invalid crop center value {} for image {}'.format(
            crop_center, rgb_file)
        if ret_width:
            return crop_center, bmax - bmin
        return crop_center

    def load_rgb(self, rgb_file, flip=False, aug_blur=0.0):
        # rgb = np.array(Image.open(rgb_file))
        # if flip:
        #     rgb = self.flip_image(rgb)
        # if aug_blur > 0.000001:
        #     rgb = self.blur_image(rgb, aug_blur)
        # return rgb

        # if OLD_PATH in rgb_file:
        #     rgb_file_fast = rgb_file.replace(OLD_PATH, SCRATCH_PATH)
        #     if not osp.isfile(rgb_file_fast):
        #         rgb_file_fast = rgb_file
        # else:
        #     rgb_file_fast = rgb_file
        # else:
        #     print('fast loading RGB images')

        rgb_file_fast = rgb_file
        rgb = np.array(Image.open(rgb_file_fast))
        return rgb

    def blur_image(self, img, aug_blur):
        assert isinstance(img, np.ndarray)
        x = np.random.uniform(0, aug_blur) * 255.  # input image is in range [0, 255]
        blur = GaussianBlur(x)
        img = Image.fromarray(img)
        return np.array(img.filter(blur))
        # if self.aug_blur > 0.000001:

        return img

    def crop(self, img, center, crop_size):
        """
        crop image around the given center, pad zeros for boraders
        :param img:
        :param center:
        :param crop_size: size of the resulting crop
        :return: a square crop around the center
        """
        assert isinstance(img, np.ndarray)
        h, w = img.shape[:2]
        topleft = np.round(center - crop_size / 2).astype(int)
        bottom_right = np.round(center + crop_size / 2).astype(int)

        x1 = max(0, topleft[0])
        y1 = max(0, topleft[1])
        x2 = min(w - 1, bottom_right[0])
        y2 = min(h - 1, bottom_right[1])
        cropped = img[y1:y2, x1:x2]

        p1 = max(0, -topleft[0])  # padding in x, top
        p2 = max(0, -topleft[1])  # padding in y, top
        p3 = max(0, bottom_right[0] - w+1)  # padding in x, bottom
        p4 = max(0, bottom_right[1] - h+1)  # padding in y, bottom

        dim = len(img.shape)
        if dim == 3:
            padded = np.pad(cropped, [[p2, p4], [p1, p3], [0, 0]])
        elif dim == 2:
            padded = np.pad(cropped, [[p2, p4], [p1, p3]])
        else:
            raise NotImplemented
        return padded

    def resize(self, img, img_size, mode=cv2.INTER_LINEAR):
        """
        resize image to the input
        :param img:
        :param img_size: (width, height) of the target image size
        :param mode:
        :return:
        """
        h, w = img.shape[:2]
        load_ratio = 1.0 * w / h
        netin_ratio = 1.0 * img_size[0] / img_size[1]
        assert load_ratio == netin_ratio, "image aspect ration not matching, given image: {}, net input: {}".format(img.shape, img_size)
        resized = cv2.resize(img, img_size, interpolation=mode)
        return resized

    def compose_images(self, obj_mask, person_mask, rgb):
        """
        mask background out, and stack RGB, h+o masks
        :param obj_mask:
        :param person_mask:
        :param rgb:
        :return:
        """
        # assert self.input_type == 'RGBM3'
        # mask background out
        mask_comb = (person_mask > 0.5) | (obj_mask > 0.5)
        rgb = rgb * np.expand_dims(mask_comb, -1)
        images = np.dstack((rgb, person_mask, obj_mask))
        return images

    def load_mesh(self, file):
        "return None if file does not exist"
        if not osp.isfile(file):
            return None
        m = Mesh()
        m.load_from_file(file)
        return m

