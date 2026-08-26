import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from sound_processing import modify_sound, SOUND_FILES,save_spectrogram 
from train_cnn import TinyCNN 



MODEL_PATH = "sound_classifier_cnn.pth" 
IMG_SIZE = (64, 64)  
NUM_CLASSES = len(SOUND_FILES)  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = TinyCNN(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()  
    return model





def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")  
    image = transform(image).unsqueeze(0)  
    return image.to(DEVICE)


def predict_sound(spectrogram_path):
    if spectrogram_path is None or not os.path.exists(spectrogram_path):
        print(" Error: Spectrogram file not found.")
        return None

    print("Using spectrogram at:", spectrogram_path)

    image = preprocess_image(spectrogram_path)
    model = load_model()
    output = model(image)

    class_labels = ["explosion", "running", "walking", "wind"]
    predicted_class_idx = torch.argmax(output).item()
    predicted_label = class_labels[predicted_class_idx]

    print(f"🔊 Predicted Sound Class: {predicted_label}")


    return predicted_label


def batch_test_model(runs_per_class=100):
    model = load_model()
    correct_by_class = {label: 0 for label in SOUND_FILES}
    total_by_class = {label: 0 for label in SOUND_FILES}

   
    class_labels = ["explosion", "running", "walking", "wind"]

    for event_type in SOUND_FILES:

        print("\n Testing '" + event_type + "' " + str(runs_per_class) + " times..")

        for _ in range(runs_per_class):

            spectrogram_path = modify_sound(event_type)

            if spectrogram_path is None or not os.path.exists(spectrogram_path):
          
                print("Failed to generate spectrogram for " + event_type)

                continue

            image = preprocess_image(spectrogram_path)
            output = model(image)
            predicted_class_idx = torch.argmax(output).item()
            predicted_label = class_labels[predicted_class_idx]

            total_by_class[event_type] += 1
            if predicted_label == event_type:
                correct_by_class[event_type] += 1

        acc = correct_by_class[event_type] / total_by_class[event_type]
       

if __name__ == "__main__":


    batch_test_model(runs_per_class=100)
