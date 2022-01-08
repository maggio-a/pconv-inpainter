import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFilter as ImageFilter
import random
import math
import torch
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
from typing import Optional, Callable, Tuple, Any, Union

from . import alpha


class MaskGenerator:

    def __init__(self, items: int):
        super(MaskGenerator, self).__init__()
        self.items = items

    def generate_mask(self, height: int, width: int):
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        size = int(0.03 * (width + height))

        #for i in range(random.randint(1, self.items)):
        #    x0, y0 = random.randint(0, width - 1), random.randint(0, height)
        #    s = random.randint(3, size * 10)
        #    d = 2 * int(math.sqrt(s))
        #    xd, yd = x0 + random.randint(-d, d), y0 + random.randint(-d, d)
        #    draw.ellipse([x0, y0, x0 + s, y0 + s], fill='black')
        #    draw.ellipse([xd, yd, xd + s, yd + s], fill='white')

        for i in range(random.randint(1, self.items)):
            xa, ya = random.randint(0, width - 1), random.randint(0, height)
            xb, yb = random.randint(0, width - 1), random.randint(0, height)
            x0, y0 = min(xa, xb), min(ya, yb)
            x1, y1 = max(xa, xb), max(ya, yb)
            start_angle = random.randint(1, 180)
            end_angle = start_angle + random.randint(1, 180)
            draw.arc([x0, y0, x1, y1], start=start_angle, end=end_angle,
                     fill='black', width=random.randint(3, size))

        for i in range(random.randint(1, self.items)):
            x0, y0 = random.randint(0, width - 1), random.randint(0, height)
            diameter = 2 * random.randint(3, size)
            draw.ellipse([x0, y0, x0 + diameter, y0 + diameter], fill='black')

        for i in range(random.randint(1, self.items)):
            x0, y0 = random.randint(0, width - 1), random.randint(0, height - 1)
            x1, y1 = random.randint(0, width - 1), random.randint(0, height - 1)
            line_width = random.randint(3, size)
            draw.line([(x0, y0), (x1, y1)], fill='black', width=line_width, joint='curve')

        # MinFilter fixes an issue with PIL arc drawing :/
        return torchvision.transforms.functional.to_tensor(img.filter(ImageFilter.MinFilter))


class MaskedImageFolder(datasets.ImageFolder):

    def __init__(self,
                 root: str,
                 mask_generator_items: int,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 is_valid_file: Optional[Callable[[str], bool]] = None
                 ):
        super(MaskedImageFolder, self).__init__(root,
                                                transform=transform,
                                                target_transform=target_transform,
                                                is_valid_file=is_valid_file)
        self.mask_generator = MaskGenerator(mask_generator_items)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, class_index = super(MaskedImageFolder, self).__getitem__(index)
        mask = self.mask_generator.generate_mask(img.shape[1], img.shape[2])
        return img * mask, mask, img


class MaskedImageFolderCenter(datasets.ImageFolder):

    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 is_valid_file: Optional[Callable[[str], bool]] = None
                 ):
        super(MaskedImageFolderCenter, self).__init__(root,
                                                      transform=transform,
                                                      target_transform=target_transform,
                                                      is_valid_file=is_valid_file)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, class_index = super(MaskedImageFolderCenter, self).__getitem__(index)
        mask = F.pad(torch.zeros((3, 56, 56)), (36, 36, 36, 36), mode='constant', value=1.0)
        assert mask.shape == (3, 128, 128)
        return img * mask, img[:, 32:-32, 32:-32]


class MaskLoader:

    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 ):
        super(MaskLoader, self).__init__()
        self.dataset = datasets.ImageFolder(root, transform=transform)
        self.loader = torch.utils.data.dataloader.DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=0)
        self.data_iterator = iter(self.loader)

    def get_mask(self) -> torch.Tensor:
        try:
            mask, category = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.loader)
            mask, category = next(self.data_iterator)
        return mask.squeeze()


class AlphaMaskLoader:

    def __init__(self,
                 csvfile: str,
                 root: str
                 ):
        super(AlphaMaskLoader, self).__init__()
        self.dataset = alpha.AlphaDatasetMaskOnly(csvfile, root, channels=3)
        self.loader = torch.utils.data.dataloader.DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=0)
        self.data_iterator = iter(self.loader)

    def get_mask(self) -> torch.Tensor:
        try:
            mask = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.loader)
            mask = next(self.data_iterator)
        return mask.squeeze()


class ImageFolderWithMaskLoader(datasets.ImageFolder):

    def __init__(self,
                 root: str,
                 mask_loader: Union[MaskLoader, AlphaMaskLoader],
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 is_valid_file: Optional[Callable[[str], bool]] = None
                 ):
        super(ImageFolderWithMaskLoader, self).__init__(root,
                                                        transform=transform,
                                                        target_transform=target_transform,
                                                        is_valid_file=is_valid_file)
        self.mask_loader = mask_loader

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, class_index = super(ImageFolderWithMaskLoader, self).__getitem__(index)
        mask = self.mask_loader.get_mask()
        return img * mask, mask, img

