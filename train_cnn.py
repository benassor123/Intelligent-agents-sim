# train_sound_cnn_pytorch.py

import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
from sound_processing import modify_sound, SOUND_FILES  

# Config
SPECTROGRAM_DIR = "spectrograms"
DATASET_DIR = "spectrogram_dataset"
NUM_CLASSES = len(SOUND_FILES)
NUM_SAMPLES_PER_CLASS = 50
IMG_SIZE = (64, 64)  # was (128, 128)
BATCH_SIZE = 32
EPOCHS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def generate_spectrogram_data():
    print("🎧 Generating spectrograms...")
    os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
    for label in SOUND_FILES:
        for _ in range(NUM_SAMPLES_PER_CLASS):
            modify_sound(label)


def organize_spectrograms_for_training():
    print("📦 Organizing spectrograms into dataset folders...")

    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR) 
    os.makedirs(DATASET_DIR)

    for file in os.listdir(SPECTROGRAM_DIR):
        for label in SOUND_FILES:
            if file.startswith(label):
                label_dir = os.path.join(DATASET_DIR, label)
                os.makedirs(label_dir, exist_ok=True)

         
                source_path = os.path.join(SPECTROGRAM_DIR, file)
                dest_path = os.path.join(label_dir, file)

                print(f"📂 Moving {source_path} → {dest_path}") 
                shutil.copy(source_path, dest_path)

    print(" Spect successfully organised.")




class TinyCNN(nn.Module):
    def __init__(self, num_classes):
        # input layer to hidden layer to output
        super(TinyCNN, self).__init__()

        self.features = nn.Sequential( # feature extractor 
            nn.Conv2d(3, 16, kernel_size=3, padding=1), #convulutional hidden layer This is a 2D convolutional layer, and it does feature extraction from images
            # 3 - the input image has 3 channels (Red, Green, Blue → RGB).
            # 16 - the conv layer will learn 16 different filters - Each one will detect a different pattern in the image (like horizontal edges, vertical lines, blobs, etc.).
            # so having 16 different filters means the output will be 16 different feature maps - a feature map is the output of a filter when applied to an image
            # A feature map is a new image that shows where that pattern exists. its like a heatmap. -  Each one lights up to show “Hey, I found something interesting here!”
            # the kernel size is the size of tha filter matrix so therefore  each filter is a 3x3 squsre
            # each filter Looks at small patches (like 3x3) at a time and Slides over the entire image piece by piece
            # the filter will slide over the image and looks at 3x3 regions at a time.
            #padding is used to helps keep the output the same size as the input.

            #remember the  original image had pixels like (R, G, B) → 3 numbers per pixel.
            #After Conv2d(3, 16, ...), the output is no longer an RGB image.
            #instead  the output is 16 separate "images" (aka feature maps),
            #each filled with 1 number per pixel that describes how strongly that filter responded to that part of the input.


            nn.ReLU(), #Rectified Linear Unit   - t’s a type of activation function.
            # takes each pixel of all 16 feature maps - keeps positives aka the important pixels  and sets negatives to zero as there not important 

            nn.MaxPool2d(2), # this is a pooling layer   with a kernel size of 2×2.


            #So just like Conv2d slides a filter over the image
            #max pooling slides a 2×2 window over the feature map and picks the maximum value inside that window.
            #this is done to shrink the feature map and keep the important things

            nn.Conv2d(16, 32, kernel_size=3, padding=1), #second convulutional layer
            # recives 16 feature maps 
            #creates 32 filters  with the same kernel size
            #this is done to pick up on more patterns in the images

            nn.ReLU(), #keep positive , set negative to zero again . we only keep the important to add non-linearity
            nn.MaxPool2d(2), # reduce the feature map again by half 
        )

        #features have now been extracted from images

        self.classifier = nn.Sequential(
            nn.Flatten(),  #takes output from the conv layer and flattens it by turning it into a 1d vector s
            #o putting the pixels from the feature map into a long line


            nn.Linear(32 * 16 * 16, 64), #now  fully connected layer 
            #Input size: 32 × 16 × 16 = 8192
           # That’s the number of features you got from the last Conv2D and MaxPool2D layers — flattened into a long 1D list.#
           # so the layer reduces those 8192 numbers into just 64 more meaningful features.
            nn.ReLU(), # out of the 64 keep positive , set negative to zero again . we only keep the important to add non-linearity
            nn.Linear(64, num_classes) #final layer! turn the 64 hidden features into a prediction
            # it will output 4 values as num_classes is 4 as theres 4 sounds we are classifying 
            # each value is a scoressaying how confident the network is for each class. 
            # its not   probabilisitic as of right now so wont add up to 1
        )

    def forward(self, x): # the method responsibiel for how data will flow through the NN - x is the input image
        x = self.features(x) #send the image through both convulutional layers pooling and relu to retrieve the detailed feature maps
        x = self.classifier(x) # now we  go through classification 
        return x # final pred




# Add this to the bottom of train_sound_cnn_pytorch.py or a new file like retrain_misclassified.py

def retrain_on_misclassified(misclassified_dir="misclassified_data"):
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    if not os.path.exists(misclassified_dir) or not any(os.listdir(misclassified_dir)):
        print(" No misclassified spectrograms to retrain on.")
        return

    print(" Retraining on misclassified spectrograms.")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(misclassified_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = TinyCNN(num_classes=len(SOUND_FILES)).to(DEVICE)
    model.load_state_dict(torch.load("sound_classifier_cnn.pth", map_location=DEVICE))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    model.train()
    for epoch in range(1):  # You can change the number of retrain epochs here
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(" Retrain Epoch:")

    torch.save(model.state_dict(), "sound_classifier_cnn.pth")
    print(" Model updated with misclassified data.")

    shutil.rmtree(misclassified_dir)
    print("🧹 Cleared misclassified spectrograms after retraining.")



def main():
    generate_spectrogram_data() # get the spectograms
    organize_spectrograms_for_training() #organises the specograms into subsets of which sound type they are

    # Data transforms
    transform = transforms.Compose([  #For every spectrogram image loaded resize it to 64×64 and convert it into a tensor that PyTorch can work with
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])


    dataset = datasets.ImageFolder(DATASET_DIR, transform=transform) #load dataa
    #print(" Class-to-Index Mapping:", dataset.class_to_idx) #what index is what class (i used this line for testing)
    train_size = int(0.8 * len(dataset)) #training data 
    val_size = len(dataset) - train_size #used to test the model so see how its learning
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) #breaks data in batches to help load into the model
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)#breaks data in batches to help load into the model

    # Initialize model
    model = TinyCNN(num_classes=NUM_CLASSES).to(DEVICE) #create the NN object 
    criterion = nn.CrossEntropyLoss() # works out how wrong the prediction is
    optimizer = optim.Adam(model.parameters(), lr=0.001) #deals with adjusting weights to perfect predictions uses adam algorithm

    # Training loop
    for epoch in range(EPOCHS): # each epoch is 1 full pass through the training set learning and adjusting weights in the hidden layers
        model.train() #train
        total_loss = 0 #tracks total errors across the batches
        correct = 0 #correct predictions
        for images, labels in train_loader: #loops thorugh the data one batch every iteration. images - specograms, labels - the correct sound class
            images, labels = images.to(DEVICE), labels.to(DEVICE)# moves to gpu if available as quicker otherwise use cpu for proc
            outputs = model(images) #feeds the batch of images to the model
            loss = criterion(outputs, labels) #compares prediction to actual class/true label. we need this to adjust weigh training

            optimizer.zero_grad() #clear gradients
            loss.backward() #calculate how the weights of the model contribute to the error
            optimizer.step() #applies weight updates based on the gradients

            total_loss += loss.item() # add current batchs loss to total to avg out when epoch finshes
            _, predicted = torch.max(outputs, 1) #finds the class with highest score 
            correct += (predicted == labels).sum().item() #checks if predicited is same as label i.e.e is correct += to correct var

        train_acc = correct / len(train_dataset) # training accuracy  for this epoch
        print(" Epoch " + str(epoch + 1) + "/" + str(EPOCHS) + ", Loss: " + str(round(total_loss, 4)) + ", Accuracy: " + str(round(train_acc, 4)))
        # print training acc for curr epoch run
        

    # Save model
    torch.save(model.state_dict(), "sound_classifier_cnn.pth")  # save the NN model as a .pth 
    print(" Model saved as 'sound_classifier_cnn.pth'") # debugging ,...looks good!







if __name__ == "__main__":
    main()
