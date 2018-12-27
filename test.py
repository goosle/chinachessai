import numpy as np
killedID   = -21
killedType = abs(killedID) // 10
#            1king1,2si0.5,3xiang0.5,4ma0.15,5ju0.3,6pao0.15,7bin0.25
killValue= {1:1,2:0.025,3:0.025,4:0.075,5:0.15,6:0.075,7:0.001}
v = killValue[killedType]
print(v)