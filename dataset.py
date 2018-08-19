import cv2
import numpy as np
import collections
from PIL import Image
from skimage.feature import hog
from feature_utilities import single_img_features

class ImageDataset():

    def __init__(self, img_list, labels,
                 type='train',
                 cspace='RGB',
                 orient=9,
                 pix_per_cell = 8,
                 cell_per_block = 2,
                 spatial_size=(32, 32),
                 hist_bins=32):
        super(ImageDataset, self).__init__()
        self.images    = img_list
        self.labels    = labels
        self.type      = type
        self.cspace    = cspace
        self.orient    = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins

    #https://discuss.pytorch.org/t/trying-to-iterate-through-my-custom-dataset/1909
    def get_train_item(self, index):

        image = Image.open((self.images[index]))
        image = np.asarray(image)

        hog_feature = self.get_feature_vector(image,
                                              self.cspace,
                                              self.orient,
                                              self.pix_per_cell,
                                              self.cell_per_block,
                                              self.spatial_size,
                                              self.hist_bins)
        label = self.labels[index]

        return hog_feature, label, index

    def get_test_item(self, index):
        image = Image.open((self.images[index]))
        image = np.asarray(image)

        hog_feature = self.get_feature_vector(image,
                                              self.cspace,
                                              self.orient,
                                              self.pix_per_cell,
                                              self.cell_per_block,
                                              self.spatial_size,
                                              self.hist_bins)
        label = self.labels[index]

        return hog_feature, label, index

    # Define a function to return HOG features and visualization
    def get_feature_vector(self, img, cspace, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
        feature = single_img_features(img,
                            color_space=cspace,
                            spatial_size=spatial_size,
                            hist_bins=hist_bins,
                            orient=orient,
                            pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel='ALL')
        return np.ravel(feature)

    def __getitem__(self, index):

        if self.type=='train': return self.get_train_item(index)
        if self.type=='test':  return self.get_test_item (index)

    def __len__(self):
        return len(self.images)

def default_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if elem_type.__module__ == 'numpy':
        return np.stack(batch, 0)
    elif isinstance(batch[0], float):
        return np.stack(batch, 0)
    elif isinstance(batch[0], int):
        return np.stack(batch, 0)
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


class DataLoader(object):
    def __init__(self, dataset, shuffle=True,  batch_size=1, collate_fn=default_collate, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn

        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __iter__(self):
        return DataLoaderIter(self)

    def __len__(self):
        return len(self.batch_sampler)


class BatchSampler(object):
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

class SequentialSampler(object):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(np.arange(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(object):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(np.random.permutation(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class DataLoaderIter(object):
    "Iterates once over the DataLoader's dataset, as specified by the sampler"

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.sample_iter = iter(self.batch_sampler)


    def __len__(self):
        return len(self.batch_sampler)

    def __next__(self):
        indices = next(self.sample_iter)  # may raise StopIteration
        batch = self.collate_fn([self.dataset[i] for i in indices])

        return batch

    def __iter__(self):
        return self


    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("DataLoaderIterator cannot be pickled")