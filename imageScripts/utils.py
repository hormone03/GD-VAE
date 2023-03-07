import torch.nn as nn
import torch
from torch.autograd import Variable

def train_classifier(classifier, optimizer,train_loader, use_cuda=False):

    total_loss_value = 0.0
    number_of_observations = len(train_loader.dataset)

    for i, (images, labels) in enumerate(train_loader):
        
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()

        images = Variable(images)
        labels = Variable(labels)
        
        criterion = nn.CrossEntropyLoss()
        optimizer.zero_grad()
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss_value += loss.item()

    normalized_loss = total_loss_value / number_of_observations

    return normalized_loss

def evaluate_classifier(classifier, test_loader, use_cuda=False):

    total_loss_value = 0.0
    number_of_observations = len(test_loader.dataset)

    for i, (images, labels) in enumerate(test_loader):
        
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()

        images = Variable(images)
        labels = Variable(labels)

        criterion = nn.CrossEntropyLoss()

        outputs = classifier(images)
        loss = criterion(outputs, labels)

        total_loss_value += loss.item()

    normalized_loss = total_loss_value/number_of_observations

    return normalized_loss

def return_model_accurary(classifier, test_loader, use_cuda=False):

    correct = 0
    total = 0
    for images, labels in test_loader:
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()
        images = Variable(images)
        outputs = classifier(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    return 100 * correct / total




def train_vae(svi, train_loader, use_cuda=False):

    epoch_loss = 0.

    for x, _ in train_loader:
        if use_cuda:
            x = x.cuda()
        epoch_loss += svi.step(x)

    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train

    return total_epoch_loss_train

def evaluate_vae(svi, test_loader, use_cuda=False):

    test_loss = 0.
    for x, _ in test_loader:
        if use_cuda:
            x = x.cuda()
        test_loss += svi.evaluate_loss(x)

    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test