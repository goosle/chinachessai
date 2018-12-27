# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""
import sys
import numpy as np
import copy
from configdata import *

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, actProbs):
        """Expand tree by creating new children.
        actProbs: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        #maxProb = 0 
        #selAct = 0
        for action,prob in actProbs :
            '''
            if prob > maxProb :
                maxProb = prob
                selAct = action
            '''
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)
        #return selAct

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=200):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def doMove(self,state,action):
        killedID = state.doMove(action)
        state.exchangeTurn()
        state.getAllMoves()
        return killedID
         
    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        action = 0
        killedID = 0
        testcount = 0
        while(1):
            if node.is_leaf():  
                           
                break
            testcount += 1
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            killedID = self.doMove(state,action)
            

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        actProbs, leaf_value = self._policy(state)        
        # Check for end of game.
        winner,end = state.isEnd()
        if not end:
            if killedID != 0:
                killedType = abs(killedID) // 10
                killValue= {1:1,2:0.025,3:0.025,4:0.075,5:0.15,6:0.075,7:0.001}
                v = killValue[killedType]
                leaf_value += v
                if(leaf_value>0.95):
                    leaf_value = 0.95
                #traceDebug("kill bound = %f,v=%f"%(v,leaf_value),2)
            node.expand(actProbs)
        else:
            # for end state，return the "true" leaf_value
            if winner == 0:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.curTurn else -1.0
                )

        # Update value and visit count of nodes in this traversal.
        
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        traceDebug("playout start",2)
        for n in range(self._n_playout):           
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        traceDebug("playout end",2)
        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        #act_probs1 = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        act_probs = softmax(visits)
        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=200, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def setPlayer(self, player):
        self.player = player

    def resetPlayer(self):
        self.mcts.update_with_move(-1)

    def getAction(self, state,  return_prob=False ,temp=1e-3):
        sensible_moves = state.availables        
        # the pi vector returned by MCTS as in the alphaGo Zero paper        
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(state, temp)    
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)

            if return_prob:
                move_probs = np.zeros(ACTION_PROB)
                avIndexs = state.decodeActs(acts)                 
                move_probs[avIndexs] = probs
                return move, move_probs
            else:
                return move
        else:
            traceWarn("WARNING: the board is full,can't move")

    def __str__(self):
        return "MCTS {}".format(self.player)
