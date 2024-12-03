import torch

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from data_utils import LFWUtils as LFWUtils_Linear


class FaceDataset(Dataset):
  @staticmethod
  def toTensor(pxs, norm=1):
    t = pxs.reshape(-1, LFWUtils.IMAGE_SIZE[1], LFWUtils.IMAGE_SIZE[0])
    return t / norm

  @staticmethod
  def toPixels(img):
    return img.reshape(-1)

  def __init__(self, imgs, labels, cnn_loader=False, resnet_loader=False, train_transform=None, eval_transform=None):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.imgs = imgs.to(self.device)
    self.labels = labels.to(self.device)
    self.train_transform = train_transform
    self.cnn = cnn_loader
    self.resnet = resnet_loader
    self.eval_transform = eval_transform

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    img, label = self.imgs[idx], self.labels[idx]
    if self.train_transform and torch.is_grad_enabled():
      if self.cnn or self.resnet:
        img = FaceDataset.toTensor(img, norm=255)
        img = self.train_transform(img)
        if self.resnet:
          img = torch.stack([img[0],img[0],img[0]], dim=0)
        if self.eval_transform:
          img = self.eval_transform(img)
      else:
        img = FaceDataset.toTensor(img, norm=1)
        img = self.train_transform(img)
        if self.eval_transform:
          img = self.eval_transform(img)
        img = FaceDataset.toPixels(img)
    elif self.cnn or self.resnet:
      img = FaceDataset.toTensor(img, norm=255)
      if self.resnet:
          img = torch.stack([img[0],img[0],img[0]], dim=0)
      if self.eval_transform:
          img = self.eval_transform(img)

    return img, label


class LFWUtils(LFWUtils_Linear):
  @staticmethod
  def train_test_split(test_pct=0.5, random_state=101010, cnn_loader=False, resnet_loader=False, train_transform=None, eval_transform=None):
    train, test = LFWUtils_Linear.train_test_split(test_pct=test_pct, random_state=random_state)
    train_batch_size = 128 if resnet_loader else 256

    x_train = Tensor(train["pixels"])
    y_train = Tensor(train["labels"]).long()

    x_test = Tensor(test["pixels"])
    y_test = Tensor(test["labels"]).long()

    train_dataloader = DataLoader(FaceDataset(x_train, y_train, cnn_loader, resnet_loader, train_transform, eval_transform), batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(FaceDataset(x_test, y_test, cnn_loader, resnet_loader, None, eval_transform), batch_size=512)

    return train_dataloader, test_dataloader


  @staticmethod
  def get_labels(model, dataloader):
    model.eval()
    with torch.no_grad():
      data_labels = []
      pred_labels = []
      for x, y in dataloader:
        y_pred = model(x).argmax(dim=1)
        data_labels += [l.item() for l in y]
        pred_labels += [l.item() for l in y_pred]
      return data_labels, pred_labels
