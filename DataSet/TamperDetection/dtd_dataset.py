
import os
import cv2
import lmdb
import torch
import tempfile
import numpy as np
import pickle
import six
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DocTamperDataset(Dataset):
    '''
    A basic dataloader of the inference mode

    roots: path of the LMDB file
    minq : min random compression factor, choiced from (75, 80, 85, 90)
    max_readers : max_readers of the LMDB loader
    '''

    def __init__(self, roots, minq=75, max_nums=None, max_readers=64):
        self.envs = lmdb.open(roots, max_readers=max_readers, readonly=True, lock=False, readahead=False, meminit=False)
        with self.envs.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode('utf-8')))
        if max_nums is None:
            self.max_nums = self.nSamples
        self.max_nums = min(max_nums, self.nSamples)
        self.minq = minq  # Q
        # with open('qt_table.pk', 'rb') as fpk:
        #     pks = pickle.load(fpk)
        # self.pks = {}
        # for k, v in pks.items():
        #     self.pks[k] = torch.LongTensor(v)
        # with open('pks/' + roots + '_%d.pk' % minq, 'rb') as f:  # random compression factors with the same random seed
        #     self.record = pickle.load(f)
        self.totsr = ToTensorV2()
        self.toctsr = A.Compose([A.Normalize(),
                                 ToTensorV2()])

    def __len__(self):
        return self.max_nums

    def __getitem__(self, index):
        with self.envs.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode('utf-8'))
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            im = Image.open(buf)
            lbl_key = 'label-%09d' % index
            lblbuf = txn.get(lbl_key.encode('utf-8'))
            mask = (cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0) != 0).astype(np.uint8)

            mask = self.totsr(image=mask.copy())['image']
            img = np.array(im.convert("BGR"))
            dct = cv2.dct(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[0])
            im = im.convert('RGB')
            return self.toctsr(im), np.clip(np.abs(dct), 0, 255), mask.long()