import os
from cnchess import World,saveToDisk
from configdata import *
from dataprovider import ChessDataSet
from torch.utils.data import DataLoader
from cnnet import PoliceValueNet
import multiprocessing    
from multiprocessing import Pool,Process,Queue

def producer(who,q,gameBatchNum):        
    traceInfo('product thread %d start, pid =%d'%(who,os.getpid()))            
    world = world = World(None,True,False,gpu=False)
    for i in range(gameBatchNum):
        world.startSelfPlay()
        traceInfo("producer a dataset ,batch num = %d"%(i))
        q.put(world.dataset) 
    traceInfo('product thread  %s end'%(who))

def consumerToTrain(who,q,maxBlockCount,batchSize,epochs,learn_rate,lr_multiplier):
    traceInfo('consumer thread start:%d pid =%d'%(who,os.getpid()))   
    net = PoliceValueNet(MODEL_FILE_NAME)  
    dataBlockCount = 0
    while True:        
        data  = q.get()
        traceInfo("consumer get a datablock = %d"%(dataBlockCount))
        for i in range(epochs):
            dataLoader = DataLoader(data,batch_size=batchSize,shuffle=True)
            for step,(state_batch,mcts_probs_batch,winner_batch)  in (enumerate(dataLoader)):
                #old_probs, old_v = net.policyBatch(state_batch)
                loss, entropy = net.trainStep(
                        state_batch,
                        mcts_probs_batch,
                        winner_batch,
                        learn_rate*lr_multiplier)
                traceInfo("batch = %d ,loss=%f,entropy=%f"%(step,loss,entropy))
        dataBlockCount  += 1
        if dataBlockCount >= maxBlockCount :            
            break
    traceInfo('start updating  net local parameter file')
    net.save_model(MODEL_FILE_NAME)
    traceInfo('consumer thread %d end  '%(who))

def consumerToSave(who,q,maxBlockCount):
    traceInfo('consumer thread start:%d pid =%d'%(who,os.getpid()))    
    dataBlockCount = 0
    while True:        
        data  = q.get()
        traceInfo("consumer get a datablock = %d"%(dataBlockCount))
        saveToDisk(data.winner,data.whoAmI,data.moveSteps)
        dataBlockCount  += 1
        if dataBlockCount >= maxBlockCount :            
            break
       
    traceInfo('consumer thread %d end  '%(who))
class Trainer:
    def __init__(self):      



        
        self.batchNum = 10
        self.batchSize = 64
        
        self.learn_rate = NET_LEARN_RATE
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL

    def train(self,gameRound):
        pNum = multiprocessing.cpu_count()-1
        traceInfo("thread num  = %d"%(pNum))   
        dataque = Queue()     
        pProducerList = []
        for pi in range(pNum):                    
            pProducer = Process(target = producer ,args=(pi,dataque,self.batchNum))        
            pProducer.start()             
            pProducerList.append(pProducer)
        maxBlockCount = pNum*self.batchNum
        if(gameRound%5==0):
            self.lr_multiplier *= 0.9 
        #pConsumer = Process(target = consumerToTrain ,args=(0,dataque,maxBlockCount,3,self.batchSize,self.learn_rate,self.lr_multiplier))        
        pConsumer = Process(target = consumerToSave ,args=(0,dataque,maxBlockCount))        
        
        pConsumer.start()          
              
        traceInfo("all producer and consumer thread start ")
        for p in pProducerList:
            p.join()
        traceInfo(" consumer thread start wait end")
        pConsumer.join() 
        traceInfo("all thread end")
        
          
            
def main():
    trainer = Trainer()
    gameRound = 10
    for i in range(gameRound):
        traceInfo('start train %d'%(gameRound))
        trainer.train(gameRound)

if __name__ == '__main__':
    main()