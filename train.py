import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models 
from collections import OrderedDict
import os

def get_input_args():
    """
    Retrieves and parses the command line arguments for the training script.
    """
    parser = argparse.ArgumentParser(description="Train a neural network to classify images")
    parser.add_argument('data_dir', type=str, help='Path to the dataset directory')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save the trained model checkpoint')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'vgg19'], help='Model architecture (vgg16 or vgg19)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--hidden_units', type=int, default=1024,
                        help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
    return parser.parse_args()

def main():
    """
    Main function to train the image classifier.
    """
    args = get_input_args()

    # Define device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Define transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=data_transforms['train']),
        'valid': datasets.ImageFolder(os.path.join(args.data_dir, 'valid'), transform=data_transforms['valid']),
    }

    # Using the image datasets and the transforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
    }

    # Build and train the network
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_features = 25088
    elif args.arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_features = 25088
    else:
        raise ValueError("Unsupported architecture. Please choose 'vgg16' or 'vgg19'.")

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Define a new, untrained feed-forward network as a classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, args.hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(args.hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    model.to(device)

    # Train the classifier layers using backpropagation
    print_every = 5
    steps = 0
    for epoch in range(args.epochs):
        running_loss = 0
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{args.epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                running_loss = 0
                model.train()

    # Save the checkpoint
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {'architecture': args.arch,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'epochs': args.epochs,
                  'optimizer_state': optimizer.state_dict()}

    torch.save(checkpoint, args.save_dir)
    print(f"Model checkpoint saved to {args.save_dir}")

if __name__ == '__main__':
    main()
