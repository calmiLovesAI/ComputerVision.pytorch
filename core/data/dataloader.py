from torch.utils.data import DataLoader
from core.data import transforms as T

from core.data.voc import Voc


class PublicDataLoader:
    def __init__(self, dataset_name: str, batch_size, input_size):
        """
        :param dataset_name: str, 'voc' or 'coco'
        :param batch_size:
        :param input_size: tuple or list (h, w)
        """
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.input_size = input_size

        self.train_transforms = [
            T.Resize(self.input_size),
            T.ToTensor(),
            T.ImageColorJitter(),
        ]

        # self.val_transforms = [
        #     T.Resize(self.input_size),
        #     T.ToTensor()
        # ]

    @staticmethod
    def _get_voc(train, transforms):
        return Voc(train=train, transforms=transforms)

    def __call__(self, *args, **kwargs):
        if self.dataset_name == 'voc':
            train_data = self._get_voc(True, T.Compose(self.train_transforms))
            # val_data = self._get_voc(False, T.Compose(self.val_transforms))
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            # val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
            # return train_loader, val_loader
            return train_loader
        else:
            raise ValueError(f"{self.dataset_name} is not supported")
