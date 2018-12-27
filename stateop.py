
import numpy as np
import copy
from configdata import *

    #type+number of count,left->right,bottom->top
red_obj_init_list = [11,21,22,31,32,41,42,51,52,61,62,71,72,73,74,75]
my_obj_init_pos  = [94, 93, 95 ,92 ,96 ,91, 97, 90, 98, 71,77,60,62,64,66,68]# row(0,8),col(0,9)

black_obj_init_list = [-11,-21,-22,-31,-32,-41,-42,-51,-52,-61,-62,-71,-72,-73,-74,-75]
op_obj_init_pos  = [4, 3, 5 ,2 ,6 ,1, 7, 0, 8, 21,27,30,32,34,36,38]# row(0,8),col(0,9)

def actMinorLR(act):
    start  = act // 100
    lr = start // 10
    lc = start % 10
    start = lr*10 +(8-lc)
    end  = act % 100
    lr = end // 10
    lc = end % 10
    end = lr*10 +(8-lc)
    return start*100+end     

def actMinorFB(act):
    start  = act // 100
    lr = start // 10
    lc = start % 10
    start = (9-lr)*10 +lc
    end  = act % 100
    lr = end // 10
    lc = end % 10
    end = (9-lr)*10 +lc
    return start*100+end

def  setStateToDataset(state,act,winner,dataset):
    availActs =  state.getAllMoves()
    actIndexs =  state.decodeActs(availActs)
    actIndex =  state.decodeMove(act)                
    probs = np.zeros([ACTION_PROB]) 
    probs[actIndexs]  = 0.01
    probs[actIndex] = 0.99
    value = 1 if state.curTurn == winner else -1 
    iState = state.currentState()
    
    dataset.states.append(iState)
    dataset.probs.append(probs)
    dataset.winners.append(value)

class State:
    def __init__(self,whereAmI):
        self.reset(whereAmI)

    def setPicths(self):   
        for i in range(0,10):
            for j in range(0,9):  
                objid = self.board[i][j] 
                if(objid != 0):
                    self.pitchs[objid]=i*10+j

    def reset(self,whereAmI):
        self.board =  np.zeros([BOARD_ROWS,BOARD_COLS])
        self.curTurn = RED_PLAYER_NUM
        self.whereAmI = whereAmI
        self.winner = 0
        self.round = 0
        self.availables = []
        self.pitchs = {}
        self.lastStep  = -1  
        
        if  self.whereAmI == RED_PLAYER_NUM:                     
            myobjlist =  red_obj_init_list
            opobjlist =  black_obj_init_list        
        else:    
            opobjlist =  red_obj_init_list
            myobjlist =  black_obj_init_list
        board = self.board
        for id,pos in enumerate(my_obj_init_pos):
            board[pos//10][pos%10] = myobjlist[id]

        for id,pos in enumerate(op_obj_init_pos):
            board[pos//10][pos%10] = opobjlist[id]
        self.setPicths()

    
            

    def minorLR(self):
        self.board[:,:] = 0
        for objid,objpos in self.pitchs.items():
            r = objpos // 10
            c = objpos % 10
            nc = (8- c) 
            self.pitchs[objid] = r*10+nc  
            self.board[r][nc] = objid
        if self.lastStep > 0:           
            self.lastStep = actMinorLR(self.lastStep)
            

    def minorFB(self):
        self.board[:,:] = 0
        for objid,objpos in self.pitchs.items():
            r = objpos // 10
            c = objpos % 10
            nr = (9- r) 
            self.pitchs[objid] = nr*10+c  
            self.board[nr][c] = objid
        self.whereAmI = - self.whereAmI
           
        if self.lastStep > 0:    
            self.lastStep = actMinorFB(self.lastStep)
                    

    def exchangeTurn(self):
        self.curTurn = -self.curTurn#RED_PLAYER_NUM if self.curTurn == BLACK_PLAYER_NUM else BLACK_PLAYER_NUM
 
    def kingClosedMe(self,objId):
        return objId * self.whereAmI > 0

    def objOverBoarder(self,objId,objPos):
        bKingColseMe = self.kingClosedMe(objId) 
        bChessOverMe = objPos < 50
        return (bKingColseMe and bChessOverMe)  or (not bKingColseMe and not bChessOverMe)


    def getCommonValidMoves(self,objid,objPos,objtype,mvs,blocks,rowMin,rowMax,colMin,colMax):        
            valids = []       
            for id in range(0,len(mvs)) :
                mr = objPos // 10 + mvs[id][0]
                mc = objPos %10 + mvs[id][1]   
                pos = mr*10 + mc
                
                if ( mr > rowMax) or  (mr < rowMin) or (mc >colMax ) or (mc<colMin):
                    continue
                
                movid = self.board[pos//10][pos%10]              
                movtype = 0 if movid ==0 else   RED_PLAYER_NUM if movid > 0 else BLACK_PLAYER_NUM
                if( objtype == movtype) :#has self chess
                    continue
                
                if( len(blocks)>0):
                    bPos = objPos + blocks[id]
                    if self.board[bPos//10][bPos%10] != 0: #block
                        continue
                valids.append(objPos*100+pos)
            return valids

    def getRookMoves(self,objid,objPos,objtype):
        valids = []
        def appendPos(pos):            
            movid = self.board[pos//10][pos%10]              
            movtype = 0 if movid ==0 else   RED_PLAYER_NUM if movid > 0 else BLACK_PLAYER_NUM
            if( 0 == movtype) :#0
                valids.append(objPos*100 + pos)
                return 1
            else :
                if(movtype != objtype):#kill
                    valids.append(objPos*100 + pos)
                return 0

        row = objPos // 10 * 10
        col = objPos % 10
        for pos in np.arange(objPos-10,0-1,-10):
            if( appendPos(pos) == 0):
                break
        for pos in np.arange(objPos+10,90+col+1,10):
            if( appendPos(pos) == 0):
                break
        for pos in np.arange(objPos-1,row-1,-1):
            if( appendPos(pos) == 0):
                break
        for pos in np.arange(objPos+1,row+8+1,1):
            if( appendPos(pos) == 0):
                break    
        return valids

    def getCannonMoves(self,objid,objPos,objtype):
        valids = []
        def appendPos(pos):            
            movid = self.board[pos//10][pos%10]              
            movtype = 0 if movid ==0 else   RED_PLAYER_NUM if movid > 0 else BLACK_PLAYER_NUM
            if( 0 == movtype) :#0
                valids.append(objPos*100 + pos)
                return 1
            else :
                return 0
        def appendOverPos(pos):            
            movid = self.board[pos//10][pos%10]              
            movtype = 0 if movid ==0 else   RED_PLAYER_NUM if movid > 0 else BLACK_PLAYER_NUM
            if( 0 == movtype) :#0
                return 1
            elif objtype == movtype :
                return 0
            else :
                valids.append(objPos*100 + pos) # kill
                return 0
        row = objPos // 10 * 10
        col = objPos % 10
        isOver = False
        for pos in np.arange(objPos-10,0-1,-10):
            if isOver :
                if appendOverPos(pos) == 0:
                    break
            else :
                if( appendPos(pos) == 0):
                    isOver = True
        isOver = False
        for pos in np.arange(objPos+10,90+col+1,10):
            if isOver :
                if appendOverPos(pos) == 0:
                    break
            else :
                if( appendPos(pos) == 0):
                    isOver = True
        isOver = False            
        for pos in np.arange(objPos-1,row-1,-1):
            if isOver :
                if appendOverPos(pos) == 0:
                    break
            else :
                if( appendPos(pos) == 0):
                    isOver = True
        isOver = False
        for pos in np.arange(objPos+1,row+8+1,1):
            if isOver :
                if appendOverPos(pos) == 0:
                    break
            else :
                if( appendPos(pos) == 0):
                    isOver = True   
        return valids


        
    def checkValids(self,srcid,srcPos,valids):
        def directFaceCount(pos1,pos2):
            r1 = min(pos1// 10,pos2// 10) 
            r2 = max(pos1// 10,pos2// 10)                
            cpos = pos1 % 10 
            hasChessCount = 0 
            for rpos in range(r1+1,r2):
                if self.board[rpos][cpos]!= 0:
                    hasChessCount += 1
                    if hasChessCount > 1 :
                        break
            return hasChessCount
        
        newvalids = copy.deepcopy(valids)
        kingID = 11 if self.curTurn==RED_PLAYER_NUM else -11
        kingPos = self.pitchs.get(kingID,-1)
        if( -1 == kingPos):
            return valids
        opKingID = -kingID
        opKingPos = self.pitchs.get(opKingID,-1)
        # get king  pos           
        if( srcid == kingID):
            for mov in valids :
                dstpos = mov % 100
                if dstpos % 10  == opKingPos % 10 : 
                    if directFaceCount(dstpos,opKingPos)==0:
                        newvalids.remove(mov)        
        elif  ( srcPos %10 == kingPos % 10) and(kingPos % 10 == opKingPos % 10 ):
                if directFaceCount(kingPos,opKingPos) == 1:
                    for mov in valids :
                        dstpos = mov % 100
                        if( dstpos% 10  != opKingPos % 10):
                            newvalids.remove(mov) 
        return newvalids

    def getValidMoves(self,objid,objPos,checkValids = True):
        if (objid==0):
            traceError("found a buf ,objid===0")
            return []
        absobjid = abs(objid)
        objType =  absobjid // 10
        objGroup = objid// absobjid
        objCount = absobjid % 10 
        if objType == 1 : #king
            baseLayer = 1   
            mvs = [ [0,-1] ,[0,1] ,[1,0],[-1,0] ] # left move = - ,up move = -
            if self.kingClosedMe(objid) :
                valids  = self.getCommonValidMoves(objid,objPos,objGroup,mvs,[],7,9,3,5)
            else :
                valids  = self.getCommonValidMoves(objid,objPos,objGroup,mvs,[],0,2,3,5)

        elif objType == 2 : #kingman
            mvs = [ [-1,-1],[-1,1],[1,-1],[1,1] ]  
            baseLayer = 2+objCount-1
            if  self.kingClosedMe(objid) :
                valids  = self.getCommonValidMoves(objid,objPos,objGroup,mvs,[],7,9,3,5)  
            else :
                valids  = self.getCommonValidMoves(objid,objPos,objGroup,mvs,[],0,2,3,5)

        elif objType == 3: #elphant
            mvs = [[-2,-2],[-2,2],[2,-2],[2,2]]
            blocks = [-11,-9,9,11]  
            baseLayer = 4+objCount-1
            if self.kingClosedMe(objid) :
                valids = self.getCommonValidMoves(objid,objPos,objGroup,mvs,blocks,5,9,0,8) 
            else :
                valids = self.getCommonValidMoves(objid,objPos,objGroup,mvs,blocks,0,4,0,8)
                        
        elif objType == 4 :#knight             
            baseLayer = 6+objCount-1
            mvs = [[-1,-2],[-1,2],[1,-2],[1,2],[-2,-1],[-2,1],[2,-1],[2,1]]
            blocks = [-1,1,-1,1,-10,-10,10,10]
            valids = self.getCommonValidMoves(objid,objPos,objGroup,mvs,blocks,0,9,0,8)             
        
        elif objType == 5: #rook
            baseLayer = 8+objCount-1
            valids = self.getRookMoves(objid,objPos,objGroup)

        elif objType == 6: #cannon
            baseLayer = 10+objCount-1
            valids = self.getCannonMoves(objid,objPos,objGroup)

        elif objType == 7: #pawn
            if  self.objOverBoarder(objid,objPos) :
                if self.kingClosedMe(objid) :
                    mvs = [[-1,0],[0,-1],[0,1] ]
                else :
                    mvs = [[1,0],[0,-1],[0,1] ]
            else:
                if self.kingClosedMe(objid) :
                    mvs = [[-1,0] ]
                else :
                    mvs = [ [1,0] ] 
        
            baseLayer = 12+objCount-1
            valids = self.getCommonValidMoves(objid,objPos,objGroup,mvs,[],0,9,0,8)
        else :
            traceError( 'error objtype = %d'%(objType))
        if checkValids :
            valids = self.checkValids(objid,objPos,valids)
  
        return int(baseLayer),valids

    '''
        mcts interface
    '''
    def getAllMoves(self):
        valids = []
        for objid,objPos in self.pitchs.items():
            playNum = 0 if objid ==0 else   RED_PLAYER_NUM if objid > 0 else BLACK_PLAYER_NUM
            if  playNum == self.curTurn :                    
                _,thisvalids = self.getValidMoves(objid,objPos,False)
                if(len(thisvalids)>0):
                    valids.append(thisvalids)
        if (len(valids)>0):
            valids = list(np.concatenate(valids))   
        self.availables = valids     
        return valids

    def decodeMove(self,move):
        # from xyab to [187]
        start = move//100
        end = move%100
        objid = self.board[start//10][start%10]
        piece_type = abs(objid)//10
        delta = end-start
        count = abs(objid)%10

        if piece_type == 1:  # 帅
            base = -1
            dict = {1: base+1, -1: base+2, 10: base+3, -10: base+4}
            result = dict[delta]

        elif piece_type == 2:  # 士
            base = 3
            if count==1:
                dict = {9:base+1,-9:base+2,11:base+3,-11:base+4}
            else:
                dict = {9:base+5,-9:base+6,11:base+7,-11:base+8}
            result = dict[delta]

        elif piece_type == 3:  # 象
            base = 11
            if count==1:
                dict = {18:base+1,-18:base+2,22:base+3,-22:base+4}
            else:
                dict = {18:base+5,-18:base+6,22:base+7,-22:base+8}
            result = dict[delta]


        elif piece_type == 4:  # 马
            if count==1:
                base = 19
                dict = {8:base+1,-8:base+2,12:base+3,-12:base+4,19:base+5,-19:base+6,21:base+7,-21:base+8}
            else:
                base = 27
                dict = {8:base+1,-8:base+2,12:base+3,-12:base+4,19:base+5,-19:base+6,21:base+7,-21:base+8}
            result = dict[delta]


        elif piece_type == 5:  # 车
            if count==1:
                base = 35
                dict = {1:base+1,2:base+2,3:base+3,4:base+4,5:base+5,6:base+6,7:base+7,8:base+8\
                   ,-1:base+9,-2:base+10,-3:base+11,-4:base+12,-5:base+13,-6:base+14,-7:base+15,-8:base+16\
                   ,10:base+17,20:base+18,30:base+19,40:base+20,50:base+21,60:base+22,70:base+23,80:base+24,90:base+25,\
                   -10:base+26,-20:base+27,-30:base+28,-40:base+29,-50:base+30,-60:base+31,-70:base+32,-80:base+33,-90:base+34}
            else:
                base = 69
                dict = {1:base+1,2:base+2,3:base+3,4:base+4,5:base+5,6:base+6,7:base+7,8:base+8\
                   ,-1:base+9,-2:base+10,-3:base+11,-4:base+12,-5:base+13,-6:base+14,-7:base+15,-8:base+16\
                   ,10:base+17,20:base+18,30:base+19,40:base+20,50:base+21,60:base+22,70:base+23,80:base+24,90:base+25,\
                   -10:base+26,-20:base+27,-30:base+28,-40:base+29,-50:base+30,-60:base+31,-70:base+32,-80:base+33,-90:base+34}
            
            result = dict[delta]

        elif piece_type == 6:  # 炮
            if count==1:
                base = 103
                dict = {1:base+1,2:base+2,3:base+3,4:base+4,5:base+5,6:base+6,7:base+7,8:base+8\
                   ,-1:base+9,-2:base+10,-3:base+11,-4:base+12,-5:base+13,-6:base+14,-7:base+15,-8:base+16\
                   ,10:base+17,20:base+18,30:base+19,40:base+20,50:base+21,60:base+22,70:base+23,80:base+24,90:base+25,\
                   -10:base+26,-20:base+27,-30:base+28,-40:base+29,-50:base+30,-60:base+31,-70:base+32,-80:base+33,-90:base+34}
            
            else:
                base = 137
                dict = {1:base+1,2:base+2,3:base+3,4:base+4,5:base+5,6:base+6,7:base+7,8:base+8\
                   ,-1:base+9,-2:base+10,-3:base+11,-4:base+12,-5:base+13,-6:base+14,-7:base+15,-8:base+16\
                   ,10:base+17,20:base+18,30:base+19,40:base+20,50:base+21,60:base+22,70:base+23,80:base+24,90:base+25,\
                   -10:base+26,-20:base+27,-30:base+28,-40:base+29,-50:base+30,-60:base+31,-70:base+32,-80:base+33,-90:base+34}
            
            result = dict[delta]

        elif piece_type == 7:  # 兵
            if count==1:
                base = 171
                dict = {1:base + 1,-1:base + 2,10:base + 3,-10:base + 4}
            elif count==2:
                base = 175
                dict = {1:base + 1,-1:base + 2,10:base + 3,-10:base + 4}
            elif count==3:
                base = 179
                dict = {1:base + 1,-1:base + 2,10:base + 3,-10:base + 4}
            elif count==4:
                base = 183
                dict = {1:base + 1,-1:base + 2,10:base + 3,-10:base + 4}
            else:
                base = 187
                dict = {1:base + 1,-1:base + 2,10:base + 3,-10:base + 4}
            base = 191        
            result = dict[delta]
        else:
            result = None
        
        if (result == None):
            traceError("error chess type")
        return result

    def decodeActs(self,acts):      
        dAvailables = []  
        for move in acts :
            dmove = self.decodeMove(move)
            dAvailables.append(dmove)
        return dAvailables    

    def isEnd(self):        
        result = False
        kingID = 11 if self.curTurn==RED_PLAYER_NUM else -11
        kingPos = self.pitchs.get(kingID,-1)
        opKingPos = self.pitchs.get(-kingID,-1)
        if( -1 == kingPos):
            result = True
            self.winner = -self.curTurn
            traceDebug("game over,myking is killed, winner=%d"%(self.winner),2)
            return self.winner,result
        if( -1 == opKingPos):
            result = True
            self.winner = self.curTurn
            traceDebug("game over,opking is killed, winner=%d"%(self.winner),2)
            return self.winner,result
        if self.round > 120 :
            traceDebug ("game over , round > 120 no chess was killed",2)
            self.winner = 0
            result = True
        
        return self.winner ,result 


    def doMove(self,action):
        mStart = action // 100 
        mEnd = action % 100
        
        sr = mStart // 10
        sc = mStart % 10
        dr = mEnd // 10
        dc = mEnd % 10
        srcid = self.board[sr][sc]
        dstid = self.board[dr][dc]
        self.board[sr][sc] = 0
        self.board[dr][dc] = srcid
        self.round += 1
        self.pitchs[srcid]=mEnd
        if(srcid ==0):
            traceError("found srcid == 0")
            return 0
        if(dstid != 0):
            self.killObj(dstid,mEnd)
            return dstid
        self.lastStep = action
        return 0

    def killObj(self,objid,objPos):        
        self.pitchs.pop(objid,-1)
        self.round = 0
         

    def currentState(self):
        #input  输入：
        # 0,who am i
        # 1,cur turn
        # 2,己方棋子布局
        # 3-19我方每个棋子的有效行走图，死棋全为0，共16个平面 1-16
        # 20 对方棋子布局 
        # 21 对方上次移动的位置
        # 共21个平面
        board = self.board
        inputImage = np.zeros([INPUT_LAYER,BOARD_ROWS,BOARD_COLS])
        #0 where am i
        if( self.whereAmI == RED_PLAYER_NUM ):
            inputImage[0] = 1
        else :
            inputImage[0] = 0
        #1 curTurn
        if ( self.curTurn == self.whereAmI):
           inputImage[1] =  1
        else :
           inputImage[1] = 0
        #2 the pos of me 
        if ( self.curTurn == RED_PLAYER_NUM):
            inputImage[2] = np.where(board > 0,1,0)
        else :
            inputImage[2] = np.where(board < 0,1,0)
        
        #3-19 my valid moves each chess
        for objid,objpos in self.pitchs.items() :
            playerNum = RED_PLAYER_NUM if objid > 0 else BLACK_PLAYER_NUM
            if playerNum != self.curTurn :
                continue
            layer,valids = self.getValidMoves(objid,objpos)
            for pos in valids:
                pos = pos % 100
                inputImage[2+layer][pos // 10][pos  % 10] = 1

        #对方棋子布局，
        if ( self.curTurn == RED_PLAYER_NUM):
            inputImage[19]  = np.where(board < 0,1,0)
        else :
            inputImage[19] = np.where(board > 0,1,0)
        
        #last move
        if self.lastStep == -1 :
            inputImage[20] = 0
        else :
            step = self.lastStep
            pos = step // 100
            inputImage[20][pos // 10][pos%10] = 1  
               
        return inputImage
                   