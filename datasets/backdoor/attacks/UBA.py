from datasets.backdoor import Backdoor

from omegaconf import DictConfig as Config
import numpy as np
from PIL import Image
import os
from torchvision.transforms import Compose, ToPILImage

class UBA(Backdoor):
    def __init__(self, dataset, transform: Compose, backdoor_cfg: Config, train=True, num_classes=1000):
        super(UBA, self).__init__(dataset, transform, backdoor_cfg, train)

        self.train = train

        if self.train:
            backdoor_poison_num = backdoor_cfg["backdoor_poison_num"]
        else:
            backdoor_poison_num = 50000

        cache_val = "" if self.train else "_val"

        poisoning_cache = np.load(
            os.path.join(
                '/home/isabolic/solo-learn/',
                'poisoning_cache',
                f'{backdoor_cfg["full_name"]}_{backdoor_poison_num}{cache_val}_files.npz'
            )
        )
        if self.train:
            self.poisoned_set_indices = poisoning_cache['idxs'].tolist() # napraviti konzistentntim s Backdoor
        else:
            self.poisoned_set_indices = (-poisoning_cache['idxs']).tolist()

        self.poisoned_samples = poisoning_cache['x_es']
        self.poisoned_targets = poisoning_cache['y_es'].astype(int)

        self.samples = dataset.samples
        self.targets = np.array([s[1] for s in self.samples])
        self.targets[self.poisoned_set_indices] = self.poisoned_targets
        self.targets = list(self.targets)

        self.pil_img_transform = ToPILImage()


    def poison(self, img: Image, label: int):
        return img, label

    def get_item(self, idx: int):
        # load image
        if idx in self.poisoned_set_indices or not self.train:
            img = np.load(self.poisoned_samples[self.poisoned_set_indices.index(idx)])
            img = self.pil_img_transform(img)
            label = self.poisoned_targets[self.poisoned_set_indices.index(idx)]
        else:
            path, label = self.samples[idx]
            img = Image.open(path).convert('RGB')

        img, poisoned_label = self.poison(img, label)
        if self.transform:
            img = self.transform(img)

        poisoned_label = label
        return img, poisoned_label, label, idx
