# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import functools
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if successorGameState.isWin():
            return 9999999
        elif successorGameState.isLose():
            return -9999999
        score = successorGameState.getScore()
        distanceToNearestFood = 0
        for x in newFood:
            d = manhattanDistance(newPos, x)
            if d < distanceToNearestFood or distanceToNearestFood == 0:
                distanceToNearestFood = d
        for x in newGhostStates:
            d = manhattanDistance(x.getPosition(), newPos)
            if x.scaredTimer == 0 and d < 3:
                return -999
            elif x.scaredTimer > 0:
                score += 1/d
        return score + 1/ distanceToNearestFood
        

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.returnedAction = None

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, depth, gamestate, numAgents, saveMove = True):
        if depth == 0:
            return scoreEvaluationFunction(gamestate)
        elif gamestate.isWin() or gamestate.isLose():
            return scoreEvaluationFunction(gamestate)
        elif depth % numAgents == 0:
            score = -999999
            pacmanActions = gamestate.getLegalActions(0)
            for action in pacmanActions:
                successorState = gamestate.generateSuccessor(0, action)
                curr = self.minimax(depth -1, successorState, numAgents, False)
                if curr > score:
                    score = curr
                    if saveMove:
                        self.returnedAction = action
            return score
        else:
            ghost = numAgents - depth % numAgents
            ghostActions = gamestate.getLegalActions(ghost)
            score = 999999
            for action in ghostActions:
                successorState = gamestate.generateSuccessor(ghost, action)
                curr = self.minimax(depth -1, successorState, numAgents, False)
                if curr < score:
                    score = curr
                    if saveMove:
                        self.returnedAction = action
            return score

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        numAgents = gameState.getNumAgents()
        depth = numAgents * self.depth
        self.minimax(depth, gameState, numAgents)
        return self.returnedAction
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphaBeta(self, depth, gamestate, numAgents, alpha, beta, saveMove = True):
        if depth == self.depth * numAgents:
            return scoreEvaluationFunction(gamestate)
        elif gamestate.isWin() or gamestate.isLose():
            return scoreEvaluationFunction(gamestate)
        elif depth % numAgents == 0:
            score = -999999
            pacmanActions = gamestate.getLegalActions(0)
            for action in pacmanActions:
                successorState = gamestate.generateSuccessor(0, action)
                curr = self.alphaBeta(depth +1, successorState, numAgents, alpha, beta, False)
                if curr > score:
                    score = curr
                    if saveMove:
                        self.returnedAction = action
                alpha = max(score, alpha)
                if score > beta:
                    return score
            return score
        else:
            score = 999999
            ghost = depth % numAgents
            ghostActions = gamestate.getLegalActions(ghost)
            for action in ghostActions:
                successorState = gamestate.generateSuccessor(ghost, action)
                curr = self.alphaBeta(depth +1, successorState, numAgents, alpha, beta, False)
                if curr < score:
                    score = curr
                    if saveMove:
                        self.returnedAction = action
                beta = min(beta, score)
                if score < alpha:
                    return score
            return score
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        numAgents = gameState.getNumAgents()
        self.alphaBeta(0, gameState, numAgents, float(-999999), float(999999))
        return self.returnedAction
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimax(self, depth,gamestate, numAgents, saveMove = True):
        if depth == self.depth * numAgents:
            return self.evaluationFunction(gamestate)
        elif gamestate.isWin() or gamestate.isLose():
            return self.evaluationFunction(gamestate)
        elif depth % numAgents == 0:
            score = -999999
            pacmanActions = gamestate.getLegalActions(0)
            for action in pacmanActions:
                successorState = gamestate.generateSuccessor(0, action)
                curr = self.expectimax(depth +1, successorState, numAgents, False)
                if curr > score:
                    score = curr
                    if saveMove:
                        self.returnedAction = action
            return score
        else:
            ghost = depth % numAgents
            ghostActions = gamestate.getLegalActions(ghost)
            p = 1 / len(ghostActions)
            score = 0
            for action in ghostActions:
                successorState = gamestate.generateSuccessor(ghost, action)
                curr = self.expectimax(depth +1, successorState, numAgents, False)
                score += p * curr
            return score

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        numAgents = gameState.getNumAgents()
        self.expectimax(0, gameState, numAgents)
        return self.returnedAction
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: maximize score while going to nearest pallet and eating ghosts(priority),
    if not possible then maximize score while not eaten by ghosts
    """
    score = currentGameState.getScore()
    pos = currentGameState.getPacmanPosition()
    ghosts = currentGameState.getGhostStates()
    pallets = currentGameState.getCapsules()
    foods = currentGameState.getFood().asList()
    numFoods = currentGameState.getNumFood()
    numPallets = len(pallets)
    score -= 3* numPallets
    score -= 2* numFoods
    if currentGameState.isWin() or numFoods == 0:
        return 999999 * score
    elif currentGameState.isLose():
        return -999999
    for ghost in ghosts:
        distance = manhattanDistance(pos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            score += 100/(1+distance)
        elif distance < 2:
            return -9999
    palletDis = 0
    for pallet in pallets:
        palletDis += manhattanDistance(pos, pallet)
    score -= palletDis
    foodDis = 9999
    for food in foods:
        if manhattanDistance(pos, food) < foodDis:
            foodDis = manhattanDistance(pos, food)
    score -= foodDis
    return score

# Abbreviation
better = betterEvaluationFunction
