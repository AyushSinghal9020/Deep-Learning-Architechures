import torch
import segmentation_models_pytorch as smp
import torch.nn as nn 

import tqdm

class CustomLoss(nn.Module):
    
    def __init__(self):
        
        super(CustomLoss,self).__init__()
        
        self.diceloss = smp.losses.DiceLoss(mode='binary')
        self.binloss = smp.losses.SoftBCEWithLogitsLoss(reduction = 'mean' , smooth_factor = 0.1)

    def forward(self, output, mask):
        
        output = torch.squeeze(output)
        mask = torch.squeeze(mask)
        
        dice = self.diceloss(output , mask)
        bce = self.binloss(output , mask)
        
        loss = dice * 0.7 + bce * 0.3
        
        return loss
      
class DiceCoef(nn.Module):
    
    def __init__(self, weight=None, size_average=True):
    
        super().__init__()

    def forward(self, y_pred, y_true, smooth=1.):
        
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        
        y_pred = torch.round((y_pred - y_pred.min()) / (y_pred.max() - y_pred.min()))
        
        intersection = (y_true * y_pred).sum()
        
        dice = (2.0 * intersection + smooth)/(y_true.sum() + y_pred.sum() + smooth)
        
        return dice

deep_lab_v3 = torch.hub.load('pytorch/vision:v0.10.0' , 'deeplabv3_resnet50', pretrained=True)

deep_lab_v3.train()
deep_lab_v3.cuda()

loss_func = CustomLoss()

optimizer_dlv3 = torch.optim.Adam(deep_lab_v3.parameters(), lr=0.001)

for epoch in tqdm.notebook.tqdm(range(5)):   

    torch.cuda.empty_cache()     

    train_loss = 0
    score = 0

    for data in tqdm.notebook.tqdm(dataloader ,total = len(dataloader)):
        torch.cuda.empty_cache()

        img, mask = data

        img = img.to("cuda")
        mask = mask.to("cuda")

        outputs = deep_lab_v3(img)["out"][: , -1 , : , :] 

        loss =  loss_func(outputs , mask)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
