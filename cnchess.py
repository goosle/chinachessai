import pygame as pyg
import os
import sys
import numpy as np
import random
from collections import defaultdict, deque
from stateop import *   
from mcts_pure import MCTSPlayer as MCTSPure
from MCTS import MCTSPlayer
from cnnet import PoliceValueNet
from configdata import *

 

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
        #print(self.id ,' checked set ',checked)

    def move(self,pos):
        self.pos = pos  

class World :
    def __init__(self,screen,IamRed=True):    
        self.epochs = 5  # num of train_steps for each update
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 1200 # num of simulations for each move
        self.c_puct = 5
        self.kl_targ = 0.02
        self.check_freq = 64
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        self.bufferSize = 10000
        self.gameBatchNum = 120
        self.gameBatchSize = 1
        self.batchSize = 32
        self.dataBuffer = deque(maxlen=self.bufferSize)  

        self.winner = 0
        
        self.objs = dict()
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
            self.objs[objid].loadImg()

        for id,pos in enumerate(op_obj_init_pos):
            objid = opobjlist[id]
            self.objs[objid] = ChessMan(self.whoAmI,-self.whoAmI,objid,pos) 
            self.objs[objid].loadImg()       
        
        self.bg = load_image('boardchess.gif')  
        self.screen = screen       
        self.currentChecked = None       
        self.moveSteps = []# all turn
        self.net = PoliceValueNet('./current_policy.model')  # './best_policy_4.model'
        #self.mcts = MCTSPure(self.whoAmI)
        self.mcts = MCTSPlayer(self.net.policyValue,c_puct = self.c_puct
                  ,n_playout=self.n_playout,is_selfplay=True)

              
        
        print('game begin player : %d '%(self.state.curTurn))
       


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

    def moveInBoard(self,srcObj,dstPos):
        action = srcObj.pos * 100 + dstPos
        self.state.doMove(action)
        

    def doMove(self,srcObj,dstPos):
        #print("%d  move from %d to %d"%(srcObj.id,srcObj.pos,dstPos))
        self.moveInBoard(srcObj,dstPos)
        srcObj.move(dstPos)
        self.moveSteps.append(srcObj.pos*100+dstPos) 
        
        
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
            print("game was over ")
            return False

        srcid = self.state.board[srcPos//10][srcPos%10]   
        if(srcid == 0):
            print("%d not chess"%(srcid))
            return False
        
        _,valids = self.state.getValidMoves(srcid,srcPos)
        aiPos = srcPos*100 +dstPos
        
        if aiPos not in valids:
            print("can't move from ", srcPos ,' to ' ,dstPos)
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
            print("game was over ")
            return         
        
        #net_input = self.board.decode_board()
        #action_probs, _ =[],0# self.net.policy_value([net_input])
        #self.board.next_move = self.board.get_best_move(action_probs[0]) #fetch from best probility of net
        #self.board.next_move = 0#self.mcts.get_move()  # 格式 xyab
        #move = self.board.next_move
    
        #move = self.randMove() 
        #move = self.mcts.getAction(self.state)
        move = self.netMove()  

        if(move==0):
            print("the turn can't move")
            return 
        start = move//100
        end = move%100
        
        selsrc = self.getSelect(start)
        if(None == selsrc):
            print( ' error ai src' ,start)
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
        print('destst ' , dst.pos ,' was killed by ' ,src.pos)
        if(dst.playerNum == src.playerNum):
            return False
        dst.isDead = True    
        self.objs.pop(dst.id)
        return True



    def isEnd(self):
        self.winner,result = self.state.isEnd()
        return result

    def startPlay(self):
        print('start game!')
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
                        
                        
            tclock.tick(20)
            self.update(self.screen)
    
    
    def selfPlayMove(self,act):
        movfrom = act // 100
        movto = act %100
        #print("move from %d to %d"%(movfrom,movto))
        objid = self.state.board[movfrom//10][movfrom%10]
        srcobj = self.objs.get(objid)
        if( srcobj == None):
            print("selfplaymove src object == NULL")
            return 
        #self.doMove(srcobj,movto) 
           
        self.state.doMove(act)
        srcobj.move(movto)
        self.exchangeTurn()
        self.state.getAllMoves()
        self.update(self.screen)

    def reset(self):
        self.state.reset(-self.whoAmI)
        self.mcts.resetPlayer()
        self.mcts.setPlayer(self.state.curTurn) 

    def startSelfPlay(self):
        self.reset()
        self.state.getAllMoves()
        states,actProbs,curTurns=[],[],[]
        round = 0
        while True:
            states.append(self.state.currentState())
            act,actProb = self.mcts.getAction(self.state,True)
            curTurns.append(self.state.curTurn)
            actProbs.append(actProb) 
            self.selfPlayMove(act)
            round += 1 
            isEnd,winner = self.state.isEnd()
            if not isEnd and round > 10:
                isEnd  = True
                winner = 0
                print( "game over for round > 1000")
            
            if isEnd :
                winners = np.zeros(len(curTurns))
                if (winner != 0):
                    winners[np.array(curTurns)==winner] = 1     
                    winners[np.array(curTurns)!=winner] = -1
                
                
                if winner != 0:
                    print("Game end. Winner is player:", winner)
                else:
                    print("Game end. Tie")
                
                return winner,zip(states,actProbs,winners)
            


    def collectTrainData(self,gameBatchSize=1):        
        for step in range(gameBatchSize):
            winner,playData = self.startSelfPlay()
            self.dataBuffer.extend(playData)

    def policyUpdate(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.dataBuffer, self.batchSize)
        '''
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        '''
        state_batch,mcts_probs_batch,winner_batch=list(zip(*mini_batch))
        old_probs, old_v = self.net.policyBatch(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.net.trainStep(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.net.policyBatch(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy


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
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio


    def train(self):
        self.update(self.screen)
        print("start train")
        tclock = pyg.time.Clock()
        try:
            for i in range(self.gameBatchNum):
                print("curBatch =  ",i)
                self.collectTrainData(self.gameBatchSize)
                if len(self.dataBuffer) > self.batchSize :
                    loss,entropy = self.policyUpdate()
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    #win_ratio = self.policy_evaluate()
                    self.net.save_model('./current_policy.model')
                    """ if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0 """
        except KeyboardInterrupt:
            print('\n\rquit')


def main():
    pyg.init()
    testmode = pyg.display.mode_ok(SCREENRECT.size, 0, 32)
    screen = pyg.display.set_mode(SCREENRECT.size,0,32)
    pyg.display.set_caption("hello")
     
    world = World(screen,True)
    #world.startPlay() 
    world.train()
   
main()
