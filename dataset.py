import os
import sys
import re
import six
import math
import lmdb
import json
import torch

from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class Batch_Balanced_Dataset(object):

    def __init__(self, opt):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, augumentation=True)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate, pin_memory=True)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                datum = data_loader_iter.next()
                image, text = datum[0], datum[1]
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                datum = self.dataloader_iter_list[i].next()
                image, text = datum[0], datum[1]
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError as e:
                print(e)
                pass
            except Exception as e:
                print(e)
                raise e

        assert len(balanced_batch_images) > 0
        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts


def hierarchical_dataset(root, opt, select_data='/'):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    dataset_log += '\n'
    Dataset = LmdbDataset
    if opt.db_type == 'xmlmdb':
        Dataset = XMLLmdbDataset
    elif opt.db_type == 'raw':
        Dataset = RawDataset
    for dirpath, dirnames, filenames in os.walk(root+'/'):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                dataset = Dataset(dirpath, opt)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log


class LmdbDataset(Dataset):

    def __init__(self, root, opt):

        self.root = root
        self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

            if not hasattr(self.opt, 'data_filtering_off') or self.opt.data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

                And if you want to evaluate them with the model trained with --sensitive option,
                use --sensitive and --data_filtering_off,
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key)
                    assert label is not None, label_key
                    label = label.decode('utf-8')

                    if len(label) > self.opt.batch_max_length:
                        # print(f'The length of the label is longer than max_length: length
                        # {len(label)}, {label} in dataset {self.root}')
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    out_of_char = f'[^{self.opt.character}]'
                    if re.search(out_of_char, label.lower()):
                        continue

                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.opt.rgb:
                    img = Image.open(buf).convert('RGB')  # for color image
                else:
                    img = Image.open(buf).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

            if hasattr(self.opt, 'sensitive') and not self.opt.sensitive:
                label = label.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.opt.character}]'
            label = re.sub(out_of_char, '', label)

        return (img, label)


class XMLLmdbDataset(Dataset):

    def __init__(self, root, opt, remove_nil_char=True):

        self.root = root
        self.opt = opt
        self.remove_nil_char = remove_nil_char
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('n_line'.encode()))
            self.nSamples = nSamples

            if not hasattr(self.opt, 'data_filtering_off') or self.opt.data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = range(self.nSamples)
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

                And if you want to evaluate them with the model trained with --sensitive option,
                use --sensitive and --data_filtering_off,
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    label_key = f'{index:09d}-label'.encode()
                    label = txn.get(label_key)
                    assert label is not None, label_key
                    label = label.decode('utf-8')

                    if len(label) > self.opt.batch_max_length:
                        # print(f'The length of the label is longer than max_length: length
                        # {len(label)}, {label} in dataset {self.root}')
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    out_of_char = f'[^{self.opt.character}]'
                    if re.search(out_of_char, label.lower()):
                        continue

                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label = txn.get(f'{index:09d}-label'.encode()).decode('utf-8')
            imgbuf = txn.get(f'{index:09d}-image'.encode())
            direction = txn.get(f'{index:09d}-direction'.encode()).decode('utf-8')
            cattr = txn.get(f'{index:09d}-cattrs'.encode())
            if cattr is not None:
                cattr = json.loads(cattr)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.opt.rgb:
                    img = Image.open(buf).convert('RGB')  # for color image
                else:
                    img = Image.open(buf).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

            if hasattr(self.opt, 'sensitive') and not self.opt.sensitive:
                label = label.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            if self.remove_nil_char:
                out_of_char = f'[^{self.opt.character}]'
                label = re.sub(out_of_char, '〓', label)

        data = {
            'label': label,
            'direction': direction,
            'cattrs': cattr
        }
        return (img, data)


class RawDataset(Dataset):

    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            if self.opt.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
            else:
                img = Image.open(self.image_path_list[index]).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        # if self.max_size[2] != w:  # add border Pad
        #     Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class RandomAspect(torch.nn.Module):
    def __init__(self, max_variation: int):
        super().__init__()
        self.max_variation = max_variation

    @staticmethod
    def get_params(img: torch.Tensor, max_variation: int):
        w, h = F._get_image_size(img)
        w = torch.randint(max(w - max_variation, w // 2), w + max_variation, size=(1,)).item()
        h = torch.randint(max(h - max_variation, h // 2), h + max_variation, size=(1,)).item()
        return w, h

    def forward(self, img):
        w, h = self.get_params(img, self.max_variation)
        return F.resize(img, (h, w))


class RandomPad(torch.nn.Module):
    def __init__(self, max_padding: int, fill=0, padding_mode="constant"):
        super().__init__()
        self.max_padding = max_padding
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img: torch.Tensor, max_padding: int):
        return torch.randint(0, max_padding, size=(4,)).tolist()

    def forward(self, img):
        pad = self.get_params(img, self.max_padding)
        return F.pad(img, pad, fill=self.fill, padding_mode=self.padding_mode)


class ConstantPad(torch.nn.Module):
    def __init__(self, padding: list, fill=0, padding_mode="constant"):
        super().__init__()
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        return F.pad(img, self.padding, fill=self.fill, padding_mode=self.padding_mode)


class Partially(torch.nn.Module):
    def __init__(self, target_aspect):
        super().__init__()
        self.target_aspect = target_aspect

    @staticmethod
    def get_params(length: int):
        return torch.randint(0, length, (1,)).item(), torch.randint(0, 2, (1,)).item()

    def forward(self, img, label, cattrs):
        w, h = img.size
        ll = len(cattrs)
        if ll == 0 or ll != len(label):
            pass
            # img.save(f"image_test/no_length:{label}.png")
            # print('label::::::::', label, cattrs, label)
            return img, label
        idx, way = self.get_params(ll)
        if way and 0:
            i = idx = min(idx, max(ll - 3, 0))
            _x1 = cattrs[idx]['X']
            _x2 = cattrs[idx]['X'] + cattrs[idx]['WIDTH']
            for i in reversed(range(idx, ll)):
                attr = cattrs[i]
                print(i)
                _x2 = attr['X'] + attr['WIDTH']
                asp = (_x2 - _x1) / h
                if asp <= self.target_aspect:
                    break
            print(label, label[idx:i+1], idx, i+1)
            label = label[idx:i+1]
        else:
            i = idx = max(idx, min(3, ll - 1))
            _x1 = cattrs[idx]['X']
            _x2 = cattrs[idx]['X'] + cattrs[idx]['WIDTH']
            for i, attr in enumerate(cattrs[:idx+1]):
                _x1 = attr['X']
                asp = (_x2 - _x1) / h
                if asp <= self.target_aspect:
                    break
            label = label[i:idx+1]

        # return img
        return F.crop(img, 0, _x1, h, _x2 - _x1), label


class Sideways(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, label, vert=None, cattrs=None):
        if img.width > img.height * 5 and vert == '縦':
            vert = '横'
        elif img.height > img.width * 5 and vert == '横':
            vert = '縦'
        if vert == '縦' or (label is not None and vert == '横' and len(label) == 1):
            if cattrs is not None:
                for attr in cattrs:
                    attr['X'], attr['Y'] = attr['Y'], attr['X']
                    attr['WIDTH'], attr['HEIGHT'] = attr['HEIGHT'], attr['WIDTH']
            return img.transpose(Image.ROTATE_90), label, cattrs
        elif vert == '横' or (vert == '' and len(label) == 1):
            return img, label, cattrs
        elif vert == '右から左':
            return img, label[::-1], cattrs[::-1]
        else:
            # img.save(f'image_test/{vert}-{label}.png')
            print()
            raise ValueError(f'{vert} is unknwon, {label}({len(label)})')


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, augumentation=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.aug = augumentation

    def __call__(self, batch):
        preprocess = Sideways()
        batch = [x for x in batch if x is not None]
        data = [data for _, data in batch]
        batch = [preprocess(g, data['label'], data['direction'], data['cattrs']) for g, data in batch]
        batch = list(zip(*batch))
        images, labels, cattrs = batch
        labels = list(labels)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform0 = Partially(self.imgW / self.imgH)
            transform1 = transforms.Compose([
                RandomAspect(10),
                RandomPad(10, fill=255),
                transforms.RandomAffine(degrees=2, fill=255),
            ])
            transform2 = transforms.Compose([
                NormalizePAD((input_channel, self.imgH, resized_max_w))
            ])
            transform3 = transforms.Compose([
                transforms.GaussianBlur(3, sigma=(1e-5, 0.3)),
                # transforms.Lambda(lambda g: transforms.functional.adjust_gamma(g, 0.4 + torch.rand(1) * 0.6)),
            ])

            resized_images = []
            result_labels = []
            for i, (image, cattr) in enumerate(zip(images, cattrs)):
                label = labels[i]
                plabel = label
                pimage = image

                if self.aug and cattr is not None:
                    image, label = transform0(image, label, cattr)
                    # image.save(f'./image_test/{part_label}.jpg')
                    labels[i] = label

                w, h = image.size
                ratio = w / float(h)
                resized_w0 = math.ceil(self.imgH * ratio)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                if self.aug:
                    try:
                        resized_image = image.resize((resized_w0, self.imgH), Image.BICUBIC)
                        resized_image = transform1(resized_image)
                    except ValueError as e:
                        label = plabel
                        image = pimage
                        # image.save(f"./image_test/({w},{h})({resized_w0, self.imgH}){label}.png")
                        # image.save(f"./image_test/{label}.png")
                        continue
                        raise e
                else:
                    resized_image = image

                resized_image = ConstantPad((10, 0), 255)(resized_image)
                try:
                    resized_image = resized_image.resize((resized_w, self.imgH), Image.BICUBIC)
                except ValueError as e:
                    with open('image_test/failed.txt', 'a') as f:
                        f.write(f"{label}\n")
                    # image.save(f"./image_test/{label}.png")
                    continue
                    raise e
                normalized_tensor = transform2(resized_image)
                if self.aug:
                    normalized_tensor = transform3(normalized_tensor)
                resized_images.append(normalized_tensor)
                # resized_image.save(f'./image_test/{self.aug}-{w:05d}-{label}.jpg')
                # save_image(tensor2im(normalized_tensor), f'./image_test/{self.aug}-{w:05d}-{label}.jpg')
                result_labels.append(label)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
            labels = result_labels

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels, data


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
