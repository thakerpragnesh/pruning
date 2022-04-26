#!/usr/bin/env python
# coding: utf-8

# In[1]:
#import torch library to build neural network
import torch  # Elementory function of tensor is define in torch package
import torch.nn as nn # Several layer architectur is define here
import torch.nn.functional as F # loss function and activation function

"""
Computer vision is one of the most important application and thus lots 
of deplopment in the and torch.vision provides many facilities that can 
be use to imporve model such as data augmentation, reading data batchwise, 
suffling data before each epoch and many more
"""
# import torch library related to image data processing
import torchvision # provides facilities to access image dataset
from torchvision.datasets.utils import download_url 
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torchvision import datasets, models, transforms


# In[2]:
"""#Training Process
Below code will work as a base function and provide all the important 
function like compute loss, accuracy and print result in a perticular 
formate afte each epoch. Funvtion are as follow
1. Accuracy : Computer accuracy in evalutaion mode of pytorch on given dataset for given model
2. compute_batch_loss : Compute batch loss and append the loss in the list of batch loss.
3. compute_batch_loss_acc : Compute batch loss, batch accuracy and append the loss in the list of batch loss.
4. accumulate_batch_loss_acc: Accumulate loss from the list of batch and acccuraly loss.
5. Epoch end to print the output after every epoch in proper format
"""
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1) 		# get the prediction vector
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Compute loss of the given batch and return it
def compute_batch_loss(newmodel, batch_X,batch_y):
  images = batch_X.to(device)
  labels = batch_y.to(device)
  out = newmodel(images)                  		# Generate predictions
  loss = F.cross_entropy(out, labels) 			# Calculate loss
  return loss

# Computes loss and accuracy of the given batch(Used in validation)
def compute_batch_loss_acc(newmodel, batch_X,batch_y):
    images = batch_X.to(device)
    labels = batch_y.to(device)
    out = newmodel(images)                    	# Generate predictionsin_features=4096
    loss = F.cross_entropy(out, labels)   		# Calculate loss
    acc = accuracy(out, labels)           		# Calculate accuracy
    return {'val_loss': loss, 'val_acc': acc}

# At the end of epoch accumulate all batch loss and batch accueacy    
def accumulate_batch_loss_acc(outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

def epoch_end(epoch, result):
  # Print in given format 
  # Epoch [0], last_lr: 0.00278, train_loss: 1.2862, val_loss: 1.2110, val_acc: 0.6135
  strResult = "Epoch [{}], last_lr: {:.6f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
      epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc'])
  #print(strResult)
  return strResult


# In[3]:
"""## Define Training 
Here we will evalute our model after each epoch on validation dataset using evalute method
get_lr method returnd last learning rate used in the training
Here we are using one fit cycle method in which we specify the max learning rate and learning 
rate start from 1/10th value of max_lr and slowly increases the value to max_lr for 40% of updates 
then decreases to its initial value for 40% updates and then further decreases to 1/100th of max_lr 
value to perform final fine tuning.
"""
# evalute model on given dataset using given data loader
@torch.no_grad()
# evalute model on given dataset using given data loader
def evaluate(model, data_loader):
    model.eval()
    with torch.no_grad():
      for batch_X, batch_y in data_loader:
        outputs = [compute_batch_loss_acc(model,batch_X,batch_y)]
      return accumulate_batch_loss_acc(outputs)

# Use special scheduler to change the value of learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# epoch=8, max_lr=.01, weight_decay(L2-Regu parametr)=.0001,opt_func=Adam

######### Main Function To Implement Training #################
def fit_one_cycle(dataloaders,trainDir,testDir,
                  ModelName, model, device_l=torch.device('cpu'), 
                  epochs=1, max_lr=.01, weight_decay=0, L1=0,grad_clip=None, opt_func=torch.optim.SGD,logFile=''):
    torch.cuda.empty_cache()
    global device
    device = device_l
    history = []
    # Set up cutom optimizer here we will use one cycle scheduler with max learning
    # rate given by max_lr, default optimizer is SGD but we will use ADAM, and 
    # L2 Regularization using weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len( dataloaders[trainDir]) )
    print("Training Starts")
    with open(logFile, "a") as f:
      for epoch in range(epochs):
          # Training Phase 
          model.train()  #######################
          train_losses = []
          lrs = []
          #for batch in train_loader:
          for batch_X, batch_y in dataloaders[trainDir]:
              # computer the training loss of current batch
              loss = compute_batch_loss(model,batch_X,batch_y)
              l1_crit = nn.L1Loss()
              reg_loss = 0
              for param in model.parameters():
                reg_loss += l1_crit(param,target=torch.zeros_like(param))
              loss += L1*reg_loss 
              
              train_losses.append(loss)
              loss.backward() # compute the gradient of all weights
              # Clip the gradient value to maximum allowed grad_clip value
              if grad_clip: 
                  nn.utils.clip_grad_value_(model.parameters(), grad_clip)
              optimizer.step() # Updates weights 
              # pytorch by default accumulate grade history and if we dont want it
              # we should make all previous grade value equals to zero
              optimizer.zero_grad() 
              # Record & update learning rate
              lrs.append(get_lr(optimizer))
              sched.step() # Update the learning rate
              # Compute Validation Loss and Valodation Accuracy
              result = evaluate(model, dataloaders[testDir])
              # Compute Train Loss of whole epoch i.e mean of loss of batch 
              result['train_loss'] = torch.stack(train_losses).mean().item()
              # Observe how learning rate is change by schedular
              result['lrs'] = lrs
              # print the observation of each epoch in a proper format
          
          #strResult = "Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, 
            #val_acc: {:.4f}".format(epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc'])
          strResult = epoch_end(epoch, result) 
          
          f.write(f"{ModelName}-\t{strResult}\n")
          print(strResult)
          history.append(result) # append tupple result with val_acc, vall_loss, and trin_loss
        
    return history

