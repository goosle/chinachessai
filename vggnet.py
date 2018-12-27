import torch.nn as nn
import math
import os
import numpy as np
import torch 
import torch.nn.functional as TNF
import torch.optim as TOPT
from torch.autograd import Variable
from  configdata import *

END_HEIGHT = 5
END_WIDTH = 4
END_AREA = END_HEIGHT*END_WIDTH
class VGG(nn.Module):

    def __init__(self, features, num_classes=ACTION_PROB, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * END_AREA, 4096),
            nn.Sigmoid(), 
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.Sigmoid(),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )
        self.stateV =nn.Sequential(
            nn.Linear(512 * END_AREA, 4096),
            nn.ReLU(), 
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Tanh(),
                    
        )
       
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        v = x.view(x.size(0), -1)
        probs = self.classifier(v)
        stateV = self.stateV(v) 
        return probs,stateV

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = INPUT_LAYER
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [64,64, 128,128, 256,256,256,256,'M',512,512,512,512 ]
}

def vggF(pretrained=False, **kwargs):
    if( pretrained):
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['F'],batch_norm=True),**kwargs)
    return model

class PoliceValueNet():
    ''' police-value network''' 
    def __init__(self,modelFile = None, useGPU=True):
        
        self.useGPU = useGPU
        preTrain = modelFile and os.path.exists(modelFile)
        if useGPU :
            self.net = vggF(pretrained=preTrain).cuda()
        else :
            self.net = vggF(pretrained=preTrain) 
        self.weightDecay = NET_WEIGHT_DECAY
        self.lr = NET_LEARN_RATE
        self.useAdam = True
        self.optimizer =  self.getOptimizer()
        # TOPT.Adam(self.net.parameters(),weight_decay= self.weightDecay)
          
        if preTrain: 
            traceInfo('net parameter load from %s'%(modelFile))
            netParams = torch.load(modelFile)
            self.net.load_state_dict(netParams)
    def getOptimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify 
        special optimizer
        """
        lr = self.lr
        params = []
        for key, value in dict(self.net.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': self.weightDecay}]
        if self.useAdam:
            self.optimizer = torch.optim.Adam(params)
        else:
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer

    

    def policyBatch(self,stateBatch):
        if self.useGPU:
            stateBatch  = stateBatch.type(torch.FloatTensor)
            stateBatch = Variable(stateBatch.cuda())
            actProbs,value = self.net(stateBatch)
            return  actProbs.data.cpu().numpy(),value.data.cpu().numpy()
        else:
            stateBatch = Variable(stateBatch.type(torch.FloatTensor))
            actProbs,value = self.net(stateBatch)
            return  actProbs.data.numpy(),value.data.numpy()
       
  
    def policyValue(self,state):   
            
        currentState = np.ascontiguousarray(state.currentState().reshape(-1,INPUT_LAYER,BOARD_ROWS,BOARD_COLS))
        if self.useGPU :
            probs,value = self.net(Variable(torch.from_numpy(currentState)).cuda().float())
            probs = probs.data.cpu().numpy().flatten()
            value = value.data.cpu().numpy().flatten()[0]
        else :
            probs,value = self.net(Variable(torch.from_numpy(currentState)).float())
            probs = probs.data.numpy().flatten()
            value = value.data.numpy().flatten()[0]

        avIndexs = state.decodeActs(state.availables)  
        actProbs = zip(state.availables,probs[avIndexs])
        
        return actProbs,value


    def getAction(self,state):
        actProbs,_ = self.policyValue(state)
        actprobs = list(zip(*actProbs))
        actindex = np.argmax(actprobs[1])
        print(actindex)
        act = actprobs[0][actindex]
        '''
        traceDebug('action:',1,actprobs[0])
        traceDebug('probs:',1,actprobs[1])
        traceDebug('select:',1,actindex,act)
        '''
        return act    
        

    def trainStep(self,states, mctsProbs, winners,lr):    
        stateBatch = states.type(torch.FloatTensor)   
        mctsProbs =  mctsProbs.type(torch.FloatTensor)
        winners = winners.type(torch.FloatTensor)
        if self.useGPU:
            state_batch = Variable(stateBatch.cuda())
            mcts_probs = Variable(mctsProbs.cuda())
            winner_batch = Variable(winners.cuda())
        else:
            state_batch = Variable(stateBatch)
            mcts_probs = Variable(mctsProbs)
            winner_batch = Variable(winners)
        
        self.optimizer.zero_grad()
        
        """Sets the learning rate to the given value"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        preActs, values = self.net(state_batch)
        valueLoss = TNF.mse_loss(values.view(-1),winner_batch)
       
        #policyLoss1 = TNF.nll_loss(torch.log(preActs),mcts_probs)
        policyLoss = -torch.mean(
                torch.sum(mcts_probs * torch.log(preActs), 1)
                ) 
        
        loss = valueLoss + policyLoss     
        loss.backward()    
        self.optimizer.step()
              # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(preActs * torch.log(preActs), 1)
                )  
        return loss.item(),entropy.item() 
    pass

    def get_policy_param(self):
        net_params = self.net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        traceInfo('net parameter save to %s'%(model_file))
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
     
   
if  __name__ == "__main__":
    #net = Net()
    #print(net)
    net = PoliceValueNet()