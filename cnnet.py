import torch 
import torch.nn.functional as TNF
import torch.optim as TOPT
from torch.autograd import Variable
import numpy as np
from configdata import *

VALUE_LOSS = 1
class Net(torch.nn.Module):
    ''' net module '''
    def __init__(self):
        super(Net, self).__init__()
        '''
    boards :
         n,21,10,9
    calculate methods:
         1: 3 conv 64=>128=>256
         2: action probobility:conv(4)  ,fc(4*10*9) =>fc(187) 
         3: state value:conv(2) ,fc(2*10*9)=> fc(64)=>fc(1) 
        '''
        
        #batch normal
        self.BN = torch.nn.BatchNorm2d(2,affine=True)
        #conv layer
        self.conv1 = torch.nn.Conv2d(INPUT_LAYER,64,kernel_size= 3,padding=1)
        self.conv2 = torch.nn.Conv2d(64,128,kernel_size= 3,padding=1)
        self.conv3 = torch.nn.Conv2d(128,256,kernel_size= 3,padding=1)
        #action probility
        self.actConv1 = torch.nn.Conv2d(256,4,kernel_size= 1)
        self.actProb = torch.nn.Linear(4*BOARD_AREA,ACTION_PROB)
        #state value
        self.svConv1 = torch.nn.Conv2d(256,2,kernel_size=1)
        self.svFC1 = torch.nn.Linear(2*BOARD_AREA,64)
        self.stateValue = torch.nn.Linear(64,1)
        pass
    

    def forward(self,boards):
        conv1 = TNF.relu(self.conv1(boards))
        conv2 = TNF.relu(self.conv2(conv1))
        conv3 = TNF.relu(self.conv3(conv2))
        
        actConv1 = TNF.relu(self.actConv1(conv3))
        bActConv1 =actConv1 #self.BN(actConv1)
        bActConv1Flat = bActConv1.view(-1,4*BOARD_AREA) 
        actProb = torch.softmax(self.actProb(bActConv1Flat),1)
        #actProb = TNF.softmax(self.actProb(bActConv1Flat))
        svConv1 = TNF.relu(self.svConv1(conv3))
        bsvConv1 = svConv1 #self.BN(svConv1)
        svConv1Flat = bsvConv1.view(-1,2*BOARD_AREA)
        svFC1 =   torch.sigmoid(self.svFC1(svConv1Flat))
        stateValue = torch.tanh(self.stateValue(svFC1))# state value range(-1,1) -1 lose,0 tie,1 win
        return actProb ,stateValue


class PoliceValueNet():
    ''' police-value network''' 
    def __init__(self,modelFile = None, useGPU=True):
        self.useGPU = useGPU
        if useGPU :
            self.net = Net().cuda()
        else :
            self.net = Net() 
        self.l2_const = 1e-4
        self.optimizer =   TOPT.Adam(self.net.parameters(),weight_decay= self.l2_const)

        if modelFile: 
            netParams = torch.load(modelFile)
            self.net.load_state_dict(netParams)

    def policyBatch(self,stateBatch):
        if self.useGPU:
            stateBatch = Variable(torch.FloatTensor(stateBatch).cuda())
            actProbs,value = self.net(stateBatch)
            return  actProbs.data.cpu().numpy(),value.data.cpu().numpy()
        else:
            stateBatch = Variable(torch.FloatTensor(stateBatch))
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
        act = actprobs[0][actindex]
        return act    
        

    def trainStep(self,state_batch, mcts_probs, winner_batch,lr):        
        if self.useGPU:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))
        
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
        print('policyLoss ',policyLoss)
        loss = valueLoss + policyLoss     
        loss.backward()    
        self.optimizer.step()
              # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(preActs * torch.log(preActs), 1)
                )  
        return loss.data[0],entropy.data[0] 
    pass

    def get_policy_param(self):
        net_params = self.net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
     
   
if  __name__ == "__main__":
    #net = Net()
    #print(net)
    net = PoliceValueNet()