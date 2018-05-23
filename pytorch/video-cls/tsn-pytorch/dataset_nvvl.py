import torch.utils.data as data
import time
import random
import torch.multiprocessing
from multiprocessing import Pool

from PIL import Image
import numpy as np
import nvvl
from numpy.random import randint

# TODO: Remove VideoRecord, Use NVVl to parse video frame num
class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1]) - 3

    @property
    def label(self):
        return int(self._data[2])

def test():
    return 0

class TSNDataSetNVVL(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.worker_id = None
        
        if self.modality != 'RGB':
            raise ValueError("Unsupported modality mode!")

        self._parse_list()
        self.video_reader = None
        #self.video_reader = nvvl.VideoReader(0, 'warn')

    def set_video_reader(self, worker_id):
        pid = torch.multiprocessing.current_process().pid
        print("create video reader pid:{}".format(pid))
        self.video_reader_0 = nvvl.VideoReader(0, "warn")
        self.video_reader_1 = nvvl.VideoReader(0, "warn")

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split('\t')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        process_data,label = self.nvvl_get(record, segment_indices)

        while process_data is None:
            index = randint(0,len(self.video_list)-1)
            process_data,label = self.__getitem__(index)

        return process_data,label

    def nvvl_get(self, record, indices):
        image_shape = nvvl.video_size_from_file(record.path)

        start = time.time()

        destroy = False
        if image_shape.width == 640:
            video_reader = self.video_reader_0
        elif image_shape.width == 480:
            video_reader = self.video_reader_1
        else:
            video_reader = nvvl.VideoReader(0, "warn")
            destroy = True

        tensor_imgs = video_reader.get_samples_old(record.path, indices)
        end = time.time()
        print("read time: {} {}".format(end - start, indices))

        if destroy:
            video_reader.destroy()

        images = list()
        for tensor in tensor_imgs:
            tensor_img = tensor[0].numpy().astype(np.uint8)
            img = Image.fromarray(tensor_img)
            images.append(img)

        process_data = self.transform(images)
        return process_data, record.label

    def nvvl_get_sync(self, record, indices):
        images = list()

        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                tensor_imgs = self.video_reader.get_frames(record.path, index=p, length=1)
                tensor_imgs = tensor_imgs[0].numpy().astype(np.uint8)
                seg_imgs = Image.fromarray(tensor_imgs)
                if seg_imgs is None:
                    return None,None
                images.append(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)




