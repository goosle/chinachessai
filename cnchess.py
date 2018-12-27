import pygame as pyg
import os
import sys
import numpy as np
import random
from collections import defaultdict, deque
from stateop import *   
from dataprovider import ChessDataSet
from torch.utils.data import DataLoader
from mcts_pure import MCTSPlayer as MCTSPure
from MCTS import MCTSPlayer
#from cnnet import PoliceValueNetvggnet
from vggnet import PoliceValueNet
from configdata import *
from tqdm import tqdm
import copy 

SCREENRECT = pyg.Rect(0, 0, 720, 800)
       
#load image
def load_image(file):
    "loads an image, prepares it for play"
    main_dir = os.path.split(os.path.abspath(__file__))[0]
    file = os.path.join(main_dir, 'res', file)
    try:
        surface = pyg.image.load(file)
    except pyg.error :
        raise SystemExit('Could not load image "%s" %s' %
                         (file, pyg.get_error()))
    return surface.convert()

def saveToDisk(winner,whoAmI,moveSteps):
    traceInfo('the game round was saved')
    fn = DATA_FILE_NAME
    f = open(fn,'a+')
    f.seek(0,2)
    line = moveSteps.copy()
    line.insert(0,winner)    
    line.insert(0,whoAmI)
    
    sline = ','.join(map(str,line))
    sline += '\n'
    f.write(sline)
    f.close()

class ChessMan :
    '''
    @1 playNum: 1 ='red' ,2='black'
    @2 id : object ID
    @3 pos: object position
    '''
    def __init__(self,whereAmI,playerNum,id,pos):
        self.whereAmI = whereAmI
        self.playerNum = playerNum
        self.id = id
        
        self.chessType = abs(id) // 10
        self.chessNum = abs(id) % 10
        
        self.pos = pos
        self.isDead = False
        self.checked = False

    def loadImg(self)  :    
        if self.playerNum > 0:   
            fn = str(10+self.chessType)+'.gif'
        else:
            fn = str(20+self.chessType)+'.gif'
        self.img = load_image(fn)  



    def draw(self,sufface):  
        row = self.pos // 10
        col = self.pos % 10      
        posx = col*CHESS_WIDTH
        posy = row*CHESS_HEIGHT
        sufface.blit(self.img,(posx,posy))
        if (self.checked):
            pyg.draw.rect(sufface,(255,0,0),[posx,posy,CHESS_WIDTH,CHESS_HEIGHT],3)

    def atPos(self,pos):
        return True if self.pos == pos else  False
    
    def setChecked(self,checked):
        self.checked = checked  
        

    def move(self,pos):
        self.pos = pos  

class World :
    def __init__(self,screen,IamRed=True,shown = True,gpu = True):    
        self.epochs = 3  # num of train_steps for each update
        self.learn_rate = NET_LEARN_RATE
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 512 # num of simulations for each move
        self.c_puct = 5
        self.kl_targ = 0.02
        self.check_freq = 5
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        self.gameBatchNum = 120
        self.gameBatchSize = 1
        self.batchSize = 64
         
        self.dataset = ChessDataSet()
        self.dataLoader = DataLoader(self.dataset,batch_size=self.batchSize,shuffle=True)
        self.winner = 0
        
        self.objs = dict()
        self.shown = shown
        if IamRed:   
            self.whoAmI =RED_PLAYER_NUM                  
            myobjlist =  red_obj_init_list
            opobjlist =  black_obj_init_list        
        else:   
            self.whoAmI =BLACK_PLAYER_NUM 
            opobjlist =  red_obj_init_list
            myobjlist =  black_obj_init_list

        self.state = State(self.whoAmI)
        board = self.state.board
        for id,pos in enumerate(my_obj_init_pos):
            objid = myobjlist[id]
            self.objs[objid] = ChessMan(self.whoAmI,self.whoAmI,objid,pos)
            if shown :
                self.objs[objid].loadImg()

        for id,pos in enumerate(op_obj_init_pos):
            objid = opobjlist[id]
            self.objs[objid] = ChessMan(self.whoAmI,-self.whoAmI,objid,pos) 
            if shown :
                self.objs[objid].loadImg()       
        if shown :
            self.bg = load_image('boardchess.gif')  
            self.screen = screen       
        self.currentChecked = None       
        self.moveSteps = []# all turn
        self.net = PoliceValueNet(MODEL_FILE_NAME,useGPU = gpu)  # './best_policy_4.model'
        #self.mcts = MCTSPure(self.whoAmI,n_playout=self.n_playout)
        self.mcts = MCTSPlayer(self.net.policyValue,c_puct = self.c_puct,n_playout=self.n_playout,is_selfplay=True)
       
        traceInfo('game world init complete,player : %d '%(self.state.curTurn))
       
    def minorLR(self):
        self.state.minorLR()
        for objid,objpos in self.state.pitchs.items():
            self.objs[objid].pos = objpos
        self.update(self.screen)

    def minorFB(self):
        self.state.minorFB()
        for objid,objpos in self.state.pitchs.items():
            self.objs[objid].pos = objpos
        self.update(self.screen)


    def draw(self,sufface):
        for obj in self.objs.values():
            obj.draw(sufface)
    


    def update(self,screen):
        
        screen.blit(self.bg,(0,0)) 
        self.draw(screen)
        pyg.display.update()

    
    def exchangeTurn(self):
        self.state.exchangeTurn()


    def getSelect(self,pos):
        checked = None
        row = pos // 10
        col = pos % 10
        id = self.state.board[row,col]
        if  id == 0:            
            return None
        checked = self.objs[id]
        return checked

        

    def doMove(self,srcObj,dstPos):              
        self.moveSteps.append(srcObj.pos*100+dstPos) 
        action = srcObj.pos * 100 + dstPos
        self.state.doMove(action)
        srcObj.move(dstPos)
        
        
        
    def checked(self,pos):
        selObj = self.getSelect(pos)
        if  self.currentChecked and self.currentChecked != selObj:
            if None != selObj :
                if( self.currentChecked.playerNum == selObj.playerNum):
                    #exchagne select
                    self.currentChecked.setChecked(False)
                    self.currentChecked = None
                    selObj.setChecked(True)
                    self.currentChecked = selObj
                elif self.canMove(self.currentChecked.pos,selObj.pos):
                    self.killObj(self.currentChecked,selObj) #kill sel 
                    self.doMove(self.currentChecked,pos)
                    self.currentChecked.setChecked(False)
                    self.currentChecked = None
                    self.exchangeTurn()   
                    self.aiMove()
            else:      
                #move slect    
                if self.canMove(self.currentChecked.pos,pos):
                    self.doMove(self.currentChecked,pos)
                    self.currentChecked.setChecked(False)
                    self.currentChecked = None
                    self.exchangeTurn() 
                    self.aiMove()

        elif None != selObj and self.currentChecked != selObj:
                #set select
                if (selObj.playerNum == self.state.curTurn):
                    selObj.setChecked(True)
                    self.currentChecked  = selObj  
    


    def canMove(self,srcPos,dstPos): 
        isEnd = self.isEnd()
        if isEnd:
            traceWarn("canMove check game was over ")
            return False

        srcid = self.state.board[srcPos//10][srcPos%10]   
        if(srcid == 0):
            traceError("%d not chess"%(srcid))
            return False
        
        _,valids = self.state.getValidMoves(srcid,srcPos)
        aiPos = srcPos*100 +dstPos
        
        if aiPos not in valids:
            traceWarn("can't move from %d to %d"%( srcPos ,dstPos))
            return False   
           
        return True

       

    def randMove(self):
        valids = self.state.getAllMoves()  
        if len(valids) == 0 :
            return -1 
          
        choice = np.random.choice(valids) 
        return int(choice)  

    def netMove(self):
        self.state.getAllMoves()
        return self.net.getAction(self.state)

    def aiMove(self):
        isEnd = self.isEnd()
        if isEnd:
            traceWarn("aiMove() checking game was over ")
            return         
        move = self.netMove()  
        if(move==0):
            traceWarn("the turn can't move")
            return 
        start = move//100
        end = move%100
        
        selsrc = self.getSelect(start)
        if(None == selsrc):
            traceError( ' error ai src :%d'%(start))
            return 
        seldst = self.getSelect(end)
        if None != seldst :     
            #ai kill 
            self.killObj(selsrc,seldst)
        # ai move
        self.doMove( selsrc,end)
        #selsrc.setChecked(True)
        #self.currentChecked = selsrc
        
        self.exchangeTurn() 

    def killObj(self,src,dst):
        #to do : check can move
        #traceInfo('destst %d was killed by %d' %(dst.pos ,src.pos))
        if(dst.playerNum == src.playerNum):
            return False
        dst.isDead = True    
        self.objs.pop(dst.id)
        return True



    def isEnd(self):
        self.winner,result = self.state.isEnd()
        return result

    def startPlay(self):
        traceInfo('start game!')
        tclock = pyg.time.Clock()
        if(self.whoAmI == BLACK_PLAYER_NUM):
            self.aiMove()
        needExit = False
        while not needExit:
            for event in pyg.event.get():
                if event.type == pyg.QUIT:
                    needExit = True
                elif event.type == pyg.MOUSEBUTTONDOWN:
                    mouseTypes = pyg.mouse.get_pressed()
                    if mouseTypes[0] :
                        mx, my = pyg.mouse.get_pos()
                        row = my//CHESS_HEIGHT
                        col = mx//CHESS_WIDTH                        
                        self.checked(row*10+col)
                elif  event.type == pyg.KEYDOWN:
                    if event.unicode == 'r':
                        traceInfo("restart game") 
                        self.reset(-self.whoAmI)
                        if(self.whoAmI == BLACK_PLAYER_NUM):
                            self.aiMove()
                    elif event.unicode == 's':
                        self.save()   
                    elif event.unicode ==  'l' :
                        self.minorLR()
                    elif event.unicode ==  'f' :
                        self.minorFB()

            tclock.tick(20)
            if self.shown:
                self.update(self.screen)



    def selfPlayMove(self,act):
        movfrom = act // 100
        movto = act %100
        
        objid = self.state.board[movfrom//10][movfrom%10]
        dstid = self.state.board[movto//10][movto%10]
        
        
        srcobj = self.objs.get(objid)
        if( srcobj == None):
            traceError("selfplaymove src object == NULL")
            return         
        
        if (dstid != 0):
            dstobj = self.objs.get(dstid)
            self.killObj(srcobj,dstobj)
        
        self.doMove(srcobj,movto)
        if None != self.currentChecked:
            self.currentChecked.setChecked(False)
        srcobj.setChecked(True)
        self.currentChecked = srcobj

        self.exchangeTurn()
        self.state.getAllMoves()
        

    def reset(self,whoAmI):
        self.moveSteps.clear()
        self.whoAmI = whoAmI
        if self.whoAmI == RED_PLAYER_NUM:      
            myobjlist =  red_obj_init_list
            opobjlist =  black_obj_init_list        
        else:   
            opobjlist =  red_obj_init_list
            myobjlist =  black_obj_init_list

        
        self.objs.clear()
        for id,pos in enumerate(my_obj_init_pos):
            objid = myobjlist[id]
            self.objs[objid] = ChessMan(self.whoAmI,self.whoAmI,objid,pos)
            if self.shown:
                self.objs[objid].loadImg()

        for id,pos in enumerate(op_obj_init_pos):
            objid = opobjlist[id]
            self.objs[objid] = ChessMan(self.whoAmI,-self.whoAmI,objid,pos) 
            if self.shown:
                self.objs[objid].loadImg()   
           

        self.state.reset(self.whoAmI)
        self.mcts.resetPlayer()
        self.mcts.setPlayer(self.state.curTurn) 

    def startSelfPlay(self):
        self.reset(-self.whoAmI)
        self.state.getAllMoves()
        states,actProbs,curTurns=[],[],[]
        round = 0
        needExit = False
        while not needExit:
            states.append(self.state.currentState())
            act,actProb = self.mcts.getAction(self.state,True)
            curTurns.append(self.state.curTurn)
            actProbs.append(actProb) 
            self.selfPlayMove(act)
          
            
            if self.shown:
                for event in pyg.event.get():
                    if event.type == pyg.QUIT:
                        needExit = True
                self.update(self.screen)
            
            round += 1 
            winner,isEnd = self.state.isEnd()
            if not isEnd and round > 1000:
                isEnd  = True
                winner = 0
                traceInfo( "game over for round > 1000")
            
            if isEnd :
                winners = np.zeros(len(curTurns))
                if (winner != 0):
                    winners[np.array(curTurns)==winner] = 1     
                    winners[np.array(curTurns)!=winner] = -1
                
                
                if winner != 0:
                    traceInfo("Game end. Winner is player:%d"%(winner))
                else:
                    traceInfo("Game end. Tie")
                break
        #datas = zip(states,actProbs,winners)
        self.dataset.whoAmI = self.whoAmI
        self.dataset.winner = winner
        self.dataset.moveSteps = self.moveSteps
        for i in range(len(states)):
            self.dataset.setItem(states[i],actProbs[i],winners[i]) 
        return winner
            


    def collectTrainData(self,gameBatchSize=1):        
        for step in range(gameBatchSize):
            winner = self.startSelfPlay()
            

    def policyUpdate(self):
        """update the policy-value net"""        
        for i in range(self.epochs):
            for step,(state_batch,mcts_probs_batch,winner_batch)  in (enumerate(self.dataLoader)):
                #old_probs, old_v = self.net.policyBatch(state_batch)
                loss, entropy = self.net.trainStep(
                        state_batch,
                        mcts_probs_batch,
                        winner_batch,
                        self.learn_rate*self.lr_multiplier)
                traceInfo("loss=%f,entropy=%f"%(loss,entropy))
        self.dataset.clear()


    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.net.policy_value_fn,
                                            c_puct=self.c_puct,
                                            n_playout=self.n_playout)
        pure_mcts_player = MCTSPure(c_puct=5,
                                        n_playout=self.pure_mcts_playout_num,)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                            pure_mcts_player,
                                            start_player=i % 2,
                                            is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        traceInfo("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio


    def train(self):
        if self.shown:
            self.update(self.screen)
        traceInfo("start train")
        tclock = pyg.time.Clock()
        try:
            for i in range(self.gameBatchNum):
                traceInfo("curBatch = %d "%(i))
                self.collectTrainData(self.gameBatchSize)
                if self.gameBatchNum%5 == 0:
                        self.lr_multiplier *= 0.9
                if len(self.dataset) > self.batchSize :                    
                    self.policyUpdate()
                   
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    traceInfo("current self-play batch: {}".format(i+1))
                    #win_ratio = self.policy_evaluate()
                    self.net.save_model(MODEL_FILE_NAME)
                    """ if win_ratio > self.best_win_ratio:
                        traceInfo("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0 """
        except KeyboardInterrupt:
            traceInfo('\n\rquit')
    

    '''
    format =whoami,curturn,winner,steps
    '''  
    def save(self):
        saveToDisk(self.state.winner,self.state.curTurn,self.whoAmI,self.moveSteps)
        
   

    def trainByHis(self):
        fn = DATA_FILE_NAME
        f = open(fn,'r')
        gameCount = 0
        while  True:
            line = f.readline()
            if (None == line)or line =='':
                break
            
            gameCount += 1
            line = list(map(int,line.split(',')))
            whoAmI = line[0]
            winner = line[1]
            self.reset(whoAmI)
            acts = line[2:]
            for step,act in enumerate(acts ):   
                #source state  
                setStateToDataset(self.state,act,winner,self.dataset)            
                # left right hor minor
                mlrState = copy.deepcopy(self.state)
                mlrState.minorLR()
                mlrAct = actMinorLR(act)
                setStateToDataset(mlrState,mlrAct,winner,self.dataset) 
                #front back vertical minor
                mfbState = copy.deepcopy(self.state)
                mfbState.minorFB()
                mfbAct = actMinorFB(act)
                setStateToDataset(mfbState,mfbAct,winner,self.dataset)
                #hor and verti minor
                mlrfbState = copy.deepcopy(mlrState)
                mlrfbState.minorFB()
                mlrfbAct = actMinorFB(mlrAct)
                setStateToDataset(mlrfbState,mlrfbAct,winner,self.dataset)

                self.state.doMove(act)
                self.state.exchangeTurn()

            if gameCount%5 == 0:
                self.lr_multiplier *= 0.9
            self.policyUpdate()
            if(gameCount % 2 == 0) :
                self.net.save_model(MODEL_FILE_NAME)
            
def play():
    shown =  True
    if shown:
        pyg.init()
        testmode = pyg.display.mode_ok(SCREENRECT.size, 0, 32)
        screen = pyg.display.set_mode(SCREENRECT.size,0,32)
        pyg.display.set_caption("hello")
    else:
        screen = None 
    world = World(screen,True,shown)
    world.startPlay() 


import time
def train():
    shown =  True
    if shown:
        pyg.init()
        testmode = pyg.display.mode_ok(SCREENRECT.size, 0, 32)
        screen = pyg.display.set_mode(SCREENRECT.size,0,32)
        pyg.display.set_caption("hello")
    else:
        screen = None 
    world = World(screen,True,shown,gpu=True)    
    #world.train()
    #world.startSelfPlay()
    x = time.time()
    
    world.trainByHis()
    y = time.time()
    print('exe time = ',y-x)


if __name__ == '__main__':
    play()
    #train()
