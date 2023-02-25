import torch.optim as optim
from model import *
import util
class trainer1():
    def __init__(self,scaler, in_dim, seq_length, num_nodes, nhid , dropout, CL, model1, lrate, wdecay, device, supports,decay,clip,lr_decay_rate):
        self.model = CLDGCN(device, num_nodes, dropout, CL=CL, supports=supports, 
                           in_dim=in_dim, out_dim=seq_length, 
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)

        self.model.cuda()
        self.optimizer1 = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.CL=CL
        self.model1=model1
        # small_lr_layers = list(map(id,self.model.model.parameters()))
        # large_lr_layers = filter(lambda p:id(p) not in small_lr_layers,self.model.parameters())
        # self.optimizer = optim.Adam([
        #     {"params":large_lr_layers},
        #     {"params":self.model.model.parameters(),"lr":lrate*0.5},
        #     ],lr=lrate, weight_decay=wdecay)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = clip
        lr_decay_rate=lr_decay_rate
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: lr_decay_rate ** epoch)
        self.scheduler1 = optim.lr_scheduler.LambdaLR(self.optimizer1, lr_lambda=lambda epoch: lr_decay_rate ** epoch)
    def train(self, input, real_val):
        self.model.train()
        
        output,siam_loss,_ = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        #print(siam_loss)
        # print(real.shape)
        # print(predict.shape)
        if self.CL:
            #loss = self.loss(predict, real, 0.0)
            self.optimizer1.zero_grad()
            (siam_loss).backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer1.step() 
            loss=siam_loss
            
        else:
            loss = self.loss(predict, real, 0.0)
            self.optimizer.zero_grad()
            (loss).backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse
