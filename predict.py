import argparse
import torch
from torchvision import models, transforms
from PIL import Image
import json

def get_input_args():
    """
    Retrieves and parses the command line arguments for the prediction script.
    """
    parser = argparse.ArgumentParser(description="Predict the class of an image using a trained neural network")
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to a JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')
    return parser.parse_args()

def load_checkpoint(filepath):
    """
    Loads a checkpoint and rebuilds the model.
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    if checkpoint['architecture'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['architecture'] == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        raise ValueError("Unsupported architecture in checkpoint.")
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a tensor.
    """
    image = Image.open(image_path)
    
    # Pre-processing transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = preprocess(image)
    return image_tensor

def predict(image_path, model, topk, device):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """
    model.eval()
    model.to(device)
    
    image = process_image(image_path)
    image = image.unsqueeze(0) # Add batch dimension
    image = image.to(device)
    
    with torch.no_grad():
        output = model.forward(image)
        ps = torch.exp(output)
        top_ps, top_indices = ps.topk(topk, dim=1)
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices.cpu().numpy()[0]]
    
    return top_ps.cpu().numpy()[0], top_classes

def main():
    """
    Main function to run the prediction.
    """
    args = get_input_args()
    
    # Use GPU if available
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Load the model from checkpoint
    model = load_checkpoint(args.checkpoint)
    
    # Make prediction
    probs, classes = predict(args.image_path, model, args.top_k, device)
    
    # Load category names
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # Get class names from categories
    class_names = [cat_to_name[c] for c in classes]
    
    # Print the results
    print("\nPrediction Results:")
    for i in range(len(probs)):
        print(f"{class_names[i]:<25}: {probs[i]*100:.2f}%")

if __name__ == '__main__':
    main()
