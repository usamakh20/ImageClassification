import os
import shutil
import time
import scipy.io
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

base_dir = 'Dataset/'
images_dir = base_dir + 'Images/'
images_partition_dir = base_dir + 'Images_train_test/'
save_path = 'my_model.pth'
batch_size = 60
num_epochs = 500

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)  # 224 * 224 * 3
        self.conv2 = nn.Conv2d(32, 64, 5)  # 112 * 112 * 32
        self.conv3 = nn.Conv2d(64, 128, 3)  # 54 * 54 * 64
        self.conv4 = nn.Conv2d(128, 256, 5)  # 26 * 26 * 128
        self.conv5 = nn.Conv2d(256, 512, 3)  # 12 * 12 * 256
        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 120)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_bn1 = nn.BatchNorm2d(512)
        self.soft_max = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.conv_bn1(x)
        x = x.view(-1, 512 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.soft_max(self.fc3(x))
        return x

    def fit(self, train_loader):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        start_time = time.time()

        for epoch in range(num_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 20 == 19:  # print every 20 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 20))
                    print('Time taken: ' + str(time.time() - start_time))
                    start_time = time.time()
                    running_loss = 0.0

        print('Finished Training')

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=device))

    def test(self, test_loader):
        class_correct = list(0. for _ in range(120))
        class_total = list(0. for _ in range(120))
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                correct += (predicted == labels).sum().item()
                for i in range(batch_size):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        return correct / sum(class_total), [x / y for x, y in zip(class_correct, class_total)]


def rename(name):
    return ' '.join(' '.join(name.split('-')[1:]).split('_'))


def partition_data(dir_names):
    if not os.path.isdir(images_partition_dir):
        for option in ['train', 'test']:

            for folder in dir_names:
                os.makedirs(images_partition_dir + option + '/' + folder, exist_ok=True)

            parsed_mat_arr = scipy.io.loadmat(base_dir + 'lists/' + option + '_list.mat', squeeze_me=True)
            print('Copying ' + option + ' Images........')
            for file in parsed_mat_arr["file_list"]:
                shutil.copy(images_dir + file, images_partition_dir + option + '/' + rename(file.split('/')[0]))
            print('Finished copying.')


def get_transforms():
    return {
        'train':
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(degrees=30),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        'test':
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])}


def load_data(transforms_list):
    data_loaders = {}

    for option in ['train', 'test']:
        data_loaders[option] = \
            torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(root=images_partition_dir + option,
                                                 transform=transforms_list[option]),
                batch_size=batch_size, shuffle=True, num_workers=2
            )

    return data_loaders


if __name__ == '__main__':

    choice = input('1. Train\n2. Test\nChoose Option: ')

    if os.path.isfile('classes.txt'):
        classes = open('classes.txt', 'r').read().split('\n')[:-1]

    else:
        classes = list(map(rename, os.listdir(images_dir)))
        with open('classes.txt', 'w') as f:
            for item in classes:
                f.write("%s\n" % item)

    partition_data(classes)
    loaders = load_data(get_transforms())
    net = Net().to(device)

    if choice == '1':
        net.fit(loaders['train'])
        net.save(save_path)
    else:
        net.load(save_path)

    total_accuracy, per_class_accuracy = net.test(loaders['test'])

    print('-----------------------------------------------------------------------', end='\n\n\n')
    print('Accuracy on %d test images: %d %%' % (len(loaders['test']) * batch_size, 100 * total_accuracy))
    print('-----------------------------------------------------------------------', end='\n\n\n')
    for j in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (classes[j], 100 * per_class_accuracy[j]))
