import torchvision.transforms as transforms
from torchvision.datasets import mnist


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])

class MNIST(dict):
    def __init__(self, **kwargs):
        super(MNIST, self).__init__({
            'train': mnist.MNIST('./data', train=True, download=True, transform=transform),
            'test': mnist.MNIST('./data', train=False, download=True, transform=transform)
        })

if __name__ == '__main__':
    dataset = MNIST()
    train_dataset = dataset['test']
    img = train_dataset.__getitem__(100)[0]
    print(train_dataset.__getitem__(100)[1])
    # img = img.numpy().transpose(1, 2, 0)
    # img = (img - img.min()) / (img.max() - img.min()) * 255.0
    # img = Image.fromarray(img.astype(np.uint8))
    # img.save('test.png')