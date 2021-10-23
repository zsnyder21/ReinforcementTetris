
import numpy as np
import cv2
import torch
import random

from PIL import Image
from matplotlib import style

class Tetris(object):
    pieceColors = [
        (0, 0, 0),
        (255, 255, 0),
        (139, 81, 251),
        (49, 181, 141),
        (255, 0, 0),
        (99, 215, 240),
        (252, 147, 29),
        (0, 0, 255)
    ]

    pieces = [
        [[1, 1],
         [1, 1]],

        [[0, 2, 0],
         [2, 2, 2]],

    [[0, 3, 3],
     [3, 3, 0]],

    [[4, 4, 0],
     [0, 4, 4]],

    [[5, 5, 5, 5]],

    [[0, 0, 6],
     [6, 6, 6]],

    [[7, 0, 0],
     [7, 7, 7]]
    ]

    def __init__(self, height: int = 20, width: int = 10, blockSize: int = 20):
        self.height = height
        self.width = width
        self.blockSize = blockSize
        self.textColor = (200, 20, 220)
        self.backupBoard = np.ones(
            (self.height * self.blockSize * self.width * int(self.blockSize / 2), 3), dtype=np.uint8
        ) * np.array([201, 201, 255], dtype=np.uint8)

        self.board = None
        self.score = None
        self.tetrominoes = None
        self.clearedLines = None
        self.bag = None
        self.idx = None
        self.piece = None
        self.currentPosition = None
        self.gameOver = None

        self.reset()

    def reset(self):
        self.board = [[0] * self.width for _ in range(self.height)]
        self.score = 0
        self.tetrominoes = 0
        self.clearedLines = 0
        self.bag = list(range(len(self.pieces)))
        random.shuffle(self.bag)
        self.idx = self.bag.pop()
        self.piece = self.pieces[self.idx][:]
        self.currentPosition = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0}
        self.gameOver = False

        return self.getStateProperties(self.board)

    def getStateProperties(self, board: list) -> torch.FloatTensor:
        linesCleared, board = self.checkClearedRows(board)
        holes = self.getHoles(board)
        bumpiness, height = self.getBumpinessAndHeight(board)

        return torch.FloatTensor([linesCleared, holes, bumpiness, height])

    def checkClearedRows(self, board: list):
        toDelete = []
        for i, row in enumerate(board[::-1]):
            if 0 not in row:
                toDelete.append(len(board) - 1 - i)

        if toDelete:
            board = self.removeRow(board, toDelete)

        return len(toDelete), board

    def removeRow(self, board: list, idxs: list) -> list:
        """
        Remove filled rows from the board

        :param board: The board
        :param idxs: Indices to remove
        :return: Updated board
        """
        for i in idxs[::-1]:
            del board[i]
            board = [[0 for _ in range(self.width)]] + board

        return board

    def getHoles(self, board: list) -> int:
        """
        Count the number of holes in the board

        :param board: Board to count holes in
        :return: Number of holes
        """
        numHoles = 0
        for col in zip(*board):
            row = 0
            while row < self.height and col[row] == 0:
                row += 1

            numHoles += len([x for x in col[row + 1:] if x == 0])

        return numHoles

    def getBumpinessAndHeight(self, board: list) -> tuple:
        """
        Compute a measure of how bumpy the top of the board is and the total height

        :param board: Board to compute with
        :return: Tuple containing the bumpiness and the height
        """
        board = np.array(boarD)
        mask = board != 0
        invertedHeights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        heights = self.height - invertedHeights
        totalHeight = np.sum(heights)
        current = heights[:-1]
        next = heights[1:]
        difference = np.abs(current - next)
        totalBumpiness = np.sum(difference)

        return totalBumpiness, totalHeight

    def getNextStates(self):
        states = {}
        pieceId = self.idx
        currentPiece = self.piece[:]

        if pieceId == 0:
            rotations = 0
        elif pieceId in {2, 3, 4}:
            rotations = 2
        else:
            rotations = 4

        for i in range(rotations):
            validX = self.width - len(currentPiece[0])
            for x in range(validX + 1):
                piece = curr_piece[:]
                position = {"x": x, "y": 0}

                while not self.checkCollision(piece, position):
                    pos["y"] += 1

                self.truncate(piece, position)
                board = self.store(piece, position)
                states[(x, i)] = self.getStateProperties(board)

            currentPiece = self.rotate(currentPiece)

        return states

    def checkCollision(self, piece: list, position: dict) -> bool:
        """
        Check if a piece would collide with another piece on the board

        :param piece: Piece to check collision on
        :param position: Position of piece
        :return: Boolean indicating whether or not collision would occur
        """
        futureY = pos["y"] + 1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if futureY + y > self.height - 1 or (self.board[futureY + y][position["x"] + x] and piece[y][x]):
                    return True

        return False

    def truncate(self, piece: list, position: dict) -> bool:
        """
        Determines if the piece must be truncated and therefore if the game ends

        :param piece: Piece to check with
        :param position: Where to check
        :return: Boolean indicating whether or not the game is over
        """
        gameOver = False
        lastCollisionRow = -1

        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if self.board[position["y"] + y][position["x"] + x] and piece[y][x]:
                    if y > lastCollisionRow:
                        lastCollisionRow = y

        if position["y"] - (len(piece) - lastCollisionRow) < 0 and lastCollisionRow > -1:
            while lastCollisionRow >= 0 and len(piece) > 1:
                gameOver = True
                lastCollisionRow = -1
                del piece[0]
                for y in range(len(piece)):
                    for x in range(len(piece[y])):
                        if self.board[position["y"] + y][position["x"] + x] and piece[y][x] and y > lastCollisionRow:
                            lastCollisionRow = y

        return gameOver

    def store(self, piece: list, position: dict) -> list:
        """
        Store a newly placed piece to the board

        :param piece: Piece to store
        :param position: Where to store it
        :return: New board
        """
        board = self.board[:]
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] and not board[y + position["y"]][x + position["x"]]:
                    board[y + position["y"]][x + position["x"]] = piece[y][x]

        return board

    def rotate(self, piece: list) -> list:
        """
        Rotate a piece

        :param piece: Piece to rotate
        :return: Piece rotated by 90 degrees
        """
        rowsOriginal = columnsNew = len(piece)
        rowsNew = len(piece[0])
        rotatedPiece = []

        for i in range(rowsNew):
            newRow = [0] * columnsNew
            for j in range(columnsNew):
                newRow[j] = piece[rowsOriginal - 1 - j][i]
            rotatedPiece.append(newRow)

        return rotatedPiece

    def getCurrentBoardState(self) -> list:
        """
        Gets the current state of the board

        :return: The board
        """
        board = self.board[:]
        for y in range(len(self.piece)):
            for x in range(len(self.piece[y])):
                board[y + self.currentPosition["y"]][x + self.currentPosition["x"]] = self.piece[y][x]

        return board

    def getNewPiece(self) -> None:
        """
        Get a new piece

        :return: None
        """
        if not len(self.bag):
            self.bag = list(range(len(self.pieces)))
            random.shuffle(self.bag)

        self.idx = self.bag.pop()
        self.piece = self.pieces[self.idx][:]
        self.currentPosition = {
            "x": self.width // 2 - len(self.piece[0] // 2),
            "y": 0
        }

        if self.checkCollision(self.piece, self.currentPosition):
            self.gameOver = True

    def step(self, action: tuple, render: bool = True, video: str = None) -> tuple:
        """
        Run over a step of the game, saving necessary information happening during the step

        :param action: Action to take. A tuple of location and rotation
        :param render: Whether to render the board
        :param video: Where to save the recording?
        :return: Tuple of current score and whether or not the game is over
        """
        x, rotations = action
        self.currentPosition = {"x": x, "y": 0}

        for _ in range(rotations):
            self.piece = self.rotate(self.piece)

        while not self.checkCollision(self.piece, self.currentPosition):
            self.currentPosition["y"] += 1
            if render:
                self.render(video)

        overflow = self.truncate(self.piece, self.currentPosition)
        if overflow:
            self.gameOver = True

        self.board = self.store(self.piece, self.currentPosition)

        linesCleared, self.board = self.checkClearedRows(self.board)
        score = 1 + (linesCleared ** 2) * self.width
        self.score += score
        self.tetrominoes += 1
        self.clearedLines += linesCleared

        if not self.gameOver:
            self.getNewPiece()
        else:
            self.score -= 2

        return score, self.gameOver

    def render(self, video: str = None) -> None:
        """
        Render the game
        :param video:
        :return:
        """
        if not self.gameOver:
            img = [self.pieceColors[p] for row in self.getCurrentBoardState() for p in row]
        else:
            img = [self.pieceColors[p] for row in self.board for p in row]

        img = np.array(img).reshape((self.height, self.width, 3)).astype(np.uint8)
        img = img[..., ::-1]
        img = Image.fromarray(img, "RGB")

        img = img.resize((self.width * self.blockSize, self.height * self.blockSize))
        img = np.array(img)
        img[[i * self.blockSize for i in range(self.height)], :, :] = 0
        img[:, [i * self.blockSize for i in range(self.width)], :] = 0

        img = np.concatenate((img, self.backupBoard), axis=1)

        cv2.putText(img, "Score:", (self.width * self.blockSize + int(self.blockSize / 2), self.blockSize),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.textColor)
        cv2.putText(img, str(self.score),
                    (self.width * self.blockSize + int(self.blockSize / 2), 2 * self.blockSize),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.textColor)

        cv2.putText(img, "Pieces:", (self.width * self.block_size + int(self.block_size / 2), 4 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.textColor)
        cv2.putText(img, str(self.tetrominoes),
                    (self.width * self.blockSize + int(self.blockSize / 2), 5 * self.blockSize),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.textColor)

        cv2.putText(img, "Lines:", (self.width * self.blockSize + int(self.blockSize / 2), 7 * self.blockSize),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        cv2.putText(img, str(self.clearedLines),
                    (self.width * self.blockSize + int(self.blockSize / 2), 8 * self.blockSize),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.textColor)

        if video:
            video.write(img)

        cv2.imshow("Deep Q-Learning Tetris", img)
        cv2.waitKey(1)