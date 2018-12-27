import numpy as np
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
import configdata

class ChessDataSet(Dataset):
    def __init__(self):
        self.whoAmI = 0
        self.winner = 0
        self.moveSteps =[]

        self.states = []
        self.probs = []
        self.winners = []

    def setItem(self,state,probs,winner):
        self.states.append(state)
        self.probs.append(probs)
        self.winners.append(winner)
    def clear(self):
        self.states.clear()
        self.probs.clear()
        self.winners.clear()
        
    def __getitem__(self, index):
        return self.states[index],self.probs[index],self.winners[index]

    def __len__(self):
        return len(self.states)
def test():
    dataset = ChessDataSet()

    dataset.setItem('a',[0.88,0.76],1)
    dataset.setItem('b',[0.18,0.76],1)
    dataset.setItem('c',[0.21,0.76],1)
    dataset.setItem('e',[0.1,0.16],1)
    dataset.setItem('f',[0.2,0.26],1)
    dataset.setItem('g',[0.3,0.36],1)
    dataset.setItem('h',[0.4,0.46],1)
    dataset.setItem('i',[0.5,0.56],1)
    loader = DataLoader(dataset,batch_size=3,shuffle=True)
    for step,(s,p,w) in tqdm(enumerate(loader)):
        print(s,p,w)
        