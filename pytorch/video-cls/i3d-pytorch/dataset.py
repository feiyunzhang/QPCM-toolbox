import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class I3DDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 sample_frames=32, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, train_mode=True):

        self.root_path = root_path
        self.list_file = list_file
        self.sample_frames = sample_frames
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.train_mode = train_mode
        if not self.train_mode:
            self.num_clips = 10

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            img_path = os.path.join(directory, self.image_tmpl.format(idx))
            try:
                return [Image.open(img_path).convert('RGB')]
            except:
                print("Couldn't load image:{}".format(img_path))
                return None
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        expanded_sample_length = self.sample_frames * 2  # in order to drop every other frame
        if record.num_frames >= expanded_sample_length:
            start_pos = randint(record.num_frames - expanded_sample_length + 1)
            offsets = range(start_pos, start_pos + expanded_sample_length, 2)
        elif record.num_frames > self.sample_frames:
            average_duration = record.num_frames // self.sample_frames
            offsets = np.multiply(list(range(self.sample_frames)), average_duration) + randint(average_duration,
                                                                                            size=self.sample_frames)
        else:
            offsets = np.sort(randint(record.num_frames, size=self.sample_frames))

        offsets =[int(v)+1 for v in offsets]  # images are 1-indexed
        return offsets

    def _get_test_indices(self, record):

        def get_offsets(num_clips, num_frames_per_clip, sample_frames_per_clip):
            clip_start_pos = np.multiply(list(range(num_clips)), num_frames_per_clip)
            sample_start_pos = clip_start_pos + num_frames_per_clip // 2 - sample_frames_per_clip // 2  # temporal center of the clip
            offsets = []
            for p in sample_start_pos:
                offsets.extend(range(p, p + sample_frames_per_clip, 2))
            return offsets

        num_frames_per_clip = record.num_frames // self.num_clips
        if num_frames_per_clip >= self.sample_frames * 2: # in order to drop every other frame
            sample_frames_per_clip = self.sample_frames * 2
            offsets = get_offsets(self.num_clips,num_frames_per_clip,sample_frames_per_clip)
        elif num_frames_per_clip >= self.sample_frames:
            sample_frames_per_clip = self.sample_frames
            offsets = get_offsets(self.num_clips, num_frames_per_clip, sample_frames_per_clip)
        else:
            offsets = np.sort(randint(record.num_frames, size=self.sample_frames*self.num_clips))

        return offsets


    def __getitem__(self, index):
        record = self.video_list[index]

        if self.train_mode:
            segment_indices = self._sample_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        process_data,label = self.get(record, segment_indices)
        while process_data is None:
            index = randint(0,len(self.video_list)-1)
            process_data,label = self.__getitem__(index)
        return process_data,label

    def get(self, record, indices):

        images = list()
        for ind in indices:
            seg_img = self._load_image(record.path, ind)
            if seg_img is None:
                return None,None
            images.extend(seg_img)

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
