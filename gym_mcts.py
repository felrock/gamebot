import math
import random
import time
import copy
import mylogger

import gym
from gym import wrappers, logger

# Set flag for debug print
DEBUG = True

#####################################################################
############################ MCTSNode ###############################
#####################################################################

class MCTSNode:

    # Exploration constant
    exp = 0.05

    def __init__(self, parent, action):
        """
            Constructor for a MCTS node.
        """

        self.action           = action
        self.visits           = 0
        self.reward           = 0
        self.parent           = parent
        self.children         = []
        self.exploredChildren = 0

    def utc(self, totalVisits):
        """Calculate UTC Value."""

        if self.visits == 0:
            # Non-explored node == maximum utc score
            return math.inf

        rewardVisitsRatio = self.reward/self.visits
        explorationBias = self.exp*math.sqrt(math.log(totalVisits)/self.visits)

        return rewardVisitsRatio + explorationBias

    def childToExplore(self, totalVisits):
        """Get the best child according to UTC value."""

        bestValue = 0
        bestChild = self.children[0]

        for child in self.children:
            utcValue = child.utc(totalVisits)

            if bestValue < utcValue:
                bestValue = utcValue
                bestChild = child

        return bestChild

    def removeChild(self, child):
        """Remove child from children"""

        self.children.remove(child)

    def backpropogate(self, reward):
        """Walk up the tree and update visits and rewards"""

        if self.parent:
            self.visits += 1
            self.reward += reward
            self.parent.backpropogate(reward)
        else:
            self.visits += 1
            self.reward += reward

    def expand(self, nodes):
        """Expand current node with new nodes."""

        self.children + nodes

    def isExpanded(self):
        """Check if node is expanded"""

        return len(self.children) > 0

#####################################################################
############################ MCTSAgent ##############################
#####################################################################

class MCTSAgent(object):

    simulationDepth = 600

    def __init__(self, actions):

        self.actions = actions
        self.alen    = len(actions)

    @staticmethod
    def randomaction(env):
        """Perform a random action"""

        return env.step(env.action_space.sample())

    def mctsaction(self, env):

        # create a new tree with a cloned state
        root = MCTSNode(None, None)

        for i in range(self.simulationDepth):
            clone = env.env.clone_full_state()
            node, done = self.policy(root, env)

            if not done:
                node.children = [MCTSNode(node, action) for action in self.actions]
                random.shuffle(node.children)
            if not done:
                reward = self.rollout(env)

            node.backpropogate(reward)
            env.reset()
            env.env.restore_full_state(clone)

        bestAction = self.selectAction(root)

        # debug print
        t = root.children
        if DEBUG:
            for i in t:
                print('action {0}, reward {1:.4f}, visits {2}\n'.format(i.action, i.reward, i.visits))
        return bestAction

    def rollout(self, env):
        """Play randomly until terminal state"""
        done = False
        step = 1
        totalReward = 0

        while step != 100 and not done:

            _, reward, done, _ = env.step(env.action_space.sample())
            totalReward += (reward/step)
            step += 1

        return totalReward

    def policy(self, root, env):
        """Walk the MCT to find the optimal node to explore."""
        node = root
        done = False

        while node.children:

            if node.exploredChildren < len(node.children):
                child = node.children[node.exploredChildren]
                node.exploredChildren += 1
                node = child
            else:
                node = node.childToExplore(root.visits)

            obs, reward, done, info = env.step(node.action)

        return node, done

    def selectAction(self, node):
        """Selection action base on best reward among children"""
        action = node.children[0].action
        reward = node.children[0].reward

        for child in node.children:

            if reward <= child.reward:
                reward = child.reward
                action = child.action

        return action

#####################################################################
############################ ADD SOMETHING HERE #####################
#####################################################################
