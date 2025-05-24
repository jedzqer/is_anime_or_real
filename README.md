# is_anime_or_real
Distinguish between anime images and real images.

# You can download the model I've already trained on Hugging Face：https://huggingface.co/jedzqg/is_anime_or_real

This is based on the ResNet18 model and is used to determine whether an image is anime or real-life photography. The accuracy has limitations.
To use the model, you can refer to the following code:
        
        
      import torch
      import torch.nn as nn
      import torch.nn.functional as F
      from torchvision import transforms
      from PIL import Image
      import os
      from torchvision import models
      
      # Set up device configuration to use GPU if available
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
      # Load pre-trained ResNet18 model and modify the final layer for 2-class classification
      model = models.resnet18()
      model.fc = nn.Linear(model.fc.in_features, 2)
      model.load_state_dict(torch.load('resnet18_anime_real.pth', map_location=device))
      model.to(device)
      model.eval()
      
      # Image preprocessing pipeline matching the training setup (uses ImageNet mean/std)
      transform = transforms.Compose([
          transforms.Resize((224, 224)),  # Input size for ResNet
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
      ])
      
      # Define inference function to classify images
      def predict_image(img_path):
          image = Image.open(img_path).convert('RGB')
          image = transform(image).unsqueeze(0).to(device)  # Add batch dimension [1, 3, H, W]
      
          with torch.no_grad():  # Disable gradient calculation for inference
              output = model(image)
              predicted = torch.argmax(output, 1).item()
              # Print and return prediction results
              if predicted == 0:
                  print("✨ Prediction: This is an anime-style image! (≧∇≦)ﾉ")
                  return 0
              else:
                  print("📸 Prediction: This is a real photo! (*≧ω≦)")
                  return 1
      
      # Main execution for testing the model on image datasets
      if __name__ == '__main__':
          # Evaluate on anime images
          anime_correct = total_anime = 0
          for filename in sorted(os.listdir('./test_anime')):
              if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                  test_image_path = "./test_anime/" + filename
                  result = predict_image(test_image_path)
                  anime_correct += 1 if result == 0 else 0
                  total_anime += 1
          
          print("Starting evaluation on real photos")
          # Evaluate on real photos
          real_correct = total_real = 0
          for filename in sorted(os.listdir('./test_real')):
              if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                  test_image_path = "./test_real/" + filename
                  result = predict_image(test_image_path)
                  real_correct += 1 if result == 1 else 0
                  total_real += 1
          
          # Calculate and print accuracy metrics
          print(f"Anime image accuracy: {anime_correct}/{total_anime} = {anime_correct/total_anime:.2%}")
          print(f"Real photo accuracy: {real_correct}/{total_real} = {real_correct/total_real:.2%}")   
