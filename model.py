import torchvision.models as models
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import load_datasets
import warnings
from PIL import Image

EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 64
IMG_SIZE=224
device = torch.device("mps")

def build_model(pretrained=True, fine_tune=True, num_classes=10):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    #model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    #model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
    #model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    #model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    

    i = 0 
    j = 0
    k = 0
    for name, para in model.named_parameters():
        if para.requires_grad and 'features.0' in name:
            para.requires_grad = False 
            j += 1
        elif para.requires_grad and 'features.1' in name:
            para.requires_grad = False 
            j += 1 
        elif para.requires_grad and 'features.2' in name:
            para.requires_grad = False 
            j += 1 
        elif para.requires_grad and 'features.3' in name:
            para.requires_grad = False 
            j += 1 
        elif para.requires_grad and 'features.4' in name:
            para.requires_grad = False 
            j += 1 
        elif para.requires_grad and 'features.5' in name:
            para.requires_grad = False 
            j += 1 
        elif para.requires_grad and 'features.6' in name:
            para.requires_grad = False 
            j += 1 
        if para.requires_grad:
            k += 1
        i += 1
    #print(model)
    print(f'{j} out of {i} layers are frozen. {k} layers are trainable')
    return model

# Training function.
def train(model, trainloader, optimizer, criterion,device=torch.device("cpu")):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    i = 0
    for data in tqdm(trainloader, total=len(trainloader),ncols=20):
        i += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        counter += image.size(0)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        label_int = torch.argmax(labels, dim=1)
        step_correct = torch.eq(preds, label_int).sum().item()
        train_running_correct += step_correct
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()

        print(f'   train_loss: {train_running_loss/i:.4f}|train_acc: {train_running_correct/(counter):.4f}|correct {train_running_correct} out of {counter}', 
              end='\r')
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / i
    epoch_acc = 100. * (train_running_correct / (counter))
    return epoch_loss, epoch_acc

# Validation function.
def validate(model, testloader, criterion, device=torch.device("cpu")):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    i = 0
    with torch.no_grad():
        for data in tqdm(testloader, total=len(testloader), ncols=20):
            i += 1.0
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            counter += image.size(0)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            val_loss = criterion(outputs, labels)
            valid_running_loss += val_loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            label_int = torch.argmax(labels, dim=1)
            step_correct = torch.eq(preds, label_int).sum().item()
            valid_running_correct += step_correct
            
            print(f'   correct {valid_running_correct} out of {counter}|val_loss: {valid_running_loss/i:.4f}|val_acc: {valid_running_correct/counter:.4f}', 
                  end='\r')
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / i
    epoch_acc = 100. * (valid_running_correct / counter)
    return epoch_loss, epoch_acc

def save_model(epochs, model, optimizer, criterion, pretrained):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"model_pretrained_{pretrained}.pth")
    

def save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"accuracy_pretrained_{pretrained}.png")
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"loss_pretrained_{pretrained}.png")



def lets_go():
    efn = build_model(num_classes=264, fine_tune=False )
    train_loader, valid_loader = load_datasets(batch_size=BATCH_SIZE)
    # Optimizer.
    #optimizer = optim.Adam(efn.parameters(), lr=lr)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, efn.parameters()), lr = LR)
    # Loss function.
    criterion = nn.CrossEntropyLoss()
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     mode='min', 
                                                     factor=0.1, 
                                                     patience=2)
    best_val_loss = 1e10
    for epoch in range(EPOCHS):
        print(f"[INFO]: Epoch {epoch+1} of {EPOCHS}")
        train_epoch_loss, train_epoch_acc = train(efn.to(device), train_loader, 
                                                optimizer, criterion, device=device)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        valid_epoch_loss, valid_epoch_acc = validate(efn.to(device), valid_loader,  
                                                    criterion, device=device)
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        scheduler.step(valid_epoch_loss)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)

        print('-'*50)
        if valid_epoch_loss < best_val_loss:
            best_val_loss = valid_epoch_loss    
            # Save the trained model weights.
            save_model(EPOCHS, efn, optimizer, criterion, pretrained=True)
            print(f'Validation loss improved in epoch {epoch+1}. Saved Model')
        else:
            print(f'Validation loss did not improve in epoch {epoch+1}')
    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained=True)
    print('TRAINING COMPLETE')

def validate_model(num_classes=264):
    print('Loading model')
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    print('Getting data')
    train_loader, valid_loader = load_datasets(batch_size=4)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for data in train_loader:
            image,label = data 
            image = image.to(device)
            label = label.to(device)
            outputs = model(image)
            loss = nn.CrossEntropyLoss()
            vloss = loss(outputs, label)
            print(outputs[0], label[0])
            yhat = torch.argmax(outputs, dim=1)
            ypred = torch.argmax(label, dim=1)
            print(yhat, ypred, vloss)
            print(torch.eq(yhat, ypred).sum().item())
            break

if __name__ == "__main__":
    #validate_model()
    lets_go()
