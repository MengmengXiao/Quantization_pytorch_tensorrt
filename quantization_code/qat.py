import datetime
import os
import sys
import time
import torch
import torch.utils.data
from torch import nn
from tqdm import tqdm
import torchvision
from torchvision import transforms

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from absl import logging
logging.set_verbosity(logging.FATAL)  # Disable logging as they are too noisy in notebook
from pytorch_quantization import quant_modules
from fp32_train import *


quant_modules.initialize()
net = Net()
net.load_state_dict(torch.load("../output/quant_resnet18-calibrated.pth",map_location="cpu"))
net.cuda()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9, weight_decay=5e-4) 
    
    # Training loop
    num_epochs = 1 
    for epoch in range(num_epochs): 
        train_loss = train(net, trainloader, criterion, optimizer, device) 
        test_accuracy = test(net, testloader, criterion, device) 
    
        torch.save(net.state_dict(), "../output/qat_%d_%f.pth"%(epoch, test_accuracy))
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%") 
    
    print("Training finished")
    

