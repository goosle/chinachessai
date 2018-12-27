INPUT_LAYER = 21
BOARD_ROWS = 10
BOARD_COLS = 9
BOARD_AREA  =BOARD_ROWS * BOARD_COLS
ACTION_PROB = 192


RED_PLAYER_NUM = 1
BLACK_PLAYER_NUM = -1
CHESS_WIDTH = 80
CHESS_HEIGHT= 80   
CHESS_TYPE = [1,2,3,4,5,6,7]#King，mandarin,elephant,knight,rook,cannon,pawn

DATA_FILE_NAME='./data/1.txt'
MODEL_FILE_NAME='./model/current_policy.model'
NET_LEARN_RATE = 1e-3
NET_WEIGHT_DECAY = 1e-4

import logging
import inspect
defineLogCofig = False
if not defineLogCofig :
    defineLogCofig = True
    logging.basicConfig(level = logging.DEBUG,
    format = '%(asctime)s - %(process)d : %(message)s',
    #filename='./log/chessai.log',  
    #filemode='a'
    )

g_showLevel =2
def traceDebug(mess,showLevel,*args):
    if showLevel == g_showLevel:
        logging.debug(mess)
        if(len(args)>0):
            logging.debug(args)
    
def traceInfo(mess):
    logging.info(mess)

def traceWarn(mess):
    logging.warn(mess)

def traceError(mess):
    callerframerecord = inspect.stack()[1]
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    logging.error('%s-%d:%s'%(info.filename,info.lineno,mess))

def traceCritical(mess):
    logging.error(mess,exc_info = True)

if __name__ == '__main__':
    traceError('hah')

