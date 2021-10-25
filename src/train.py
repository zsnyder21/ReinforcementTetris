
import argparse
import os
import shutil
import numpy as np
import torch
import torch.nn as nn

from random import random, randint, sample
from tensorboardX import SummaryWriter
from Network import TetrisNetwork
from tetris import Tetris
from collections import deque


def get_args() -> argparse.Namespace:
    """
    Parse args passed from cmd

    :return: argparse.Namespace object containing passed in parameters
    """
    parser = argparse.ArgumentParser("Deep Q Network playing Tetris")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--render", default=False, help="Render the game", action=argparse.BooleanOptionalAction)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=30000)
    parser.add_argument("--save_interval", type=int, default=30)
    parser.add_argument("--replay_memory_size", type=int, default=1000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="test_trained_models")

    args = parser.parse_args()
    return args


def train(options: argparse.Namespace) -> None:
    """
    Train the network to play tetris

    :param options: argparse.Namespace object containing the training parameters
    :return: None
    """
    bestScore = 0
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    if os.path.isdir(options.log_path):
        shutil.rmtree(options.log_path)

    os.makedirs(options.log_path)
    writer = SummaryWriter(options.log_path)
    env = Tetris(width=options.width, height=options.height, blockSize=options.block_size)
    model = TetrisNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)
    criterion = nn.MSELoss()

    state = env.reset()

    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()

    replayMemory = deque(maxlen=options.replay_memory_size)
    epoch = 0

    while epoch < options.num_epochs:
        nextSteps = env.getNextStates()

        # Explore or exploit
        epsilon = options.final_epsilon + (max(options.num_decay_epochs - epoch, 0) * (
                options.initial_epsilon - options.final_epsilon) / options.num_decay_epochs)

        u = random()
        randomAction = u <= epsilon
        nextActions, nextStates = zip(*nextSteps.items())
        nextStates = torch.stack(nextStates)

        if torch.cuda.is_available():
            nextStates = nextStates.cuda()

        model.eval()

        with torch.no_grad():
            predictions = model(nextStates)[:, 0]

        model.train()

        if randomAction:
            index = randint(0, len(nextSteps) - 1)
        else:
            index = torch.argmax(predictions).item()

        nextState = nextStates[index, :]
        action = nextActions[index]

        reward, done = env.step(action, render=options.render)

        if torch.cuda.is_available():
            nextState = nextState.cuda()

        replayMemory.append([state, reward, nextState, done])

        if done:
            finalScore = env.score
            finalTetrominoes = env.tetrominoes
            finalClearedLines = env.clearedLines
            state = env.reset()

            if torch.cuda.is_available():
                state = state.cuda()
        else:
            state = nextState
            continue

        if len(replayMemory) < options.replay_memory_size / 10:
            continue

        epoch += 1
        batch = sample(replayMemory, min(len(replayMemory), options.batch_size))
        stateBatch, rewardBatch, nextStateBatch, doneBatch = zip(*batch)
        stateBatch = torch.stack(tuple(state for state in stateBatch))
        rewardBatch = torch.from_numpy(np.array(rewardBatch, dtype=np.float32)[:, None])
        nextStateBatch = torch.stack(tuple(state for state in nextStateBatch))

        if torch.cuda.is_available():
            stateBatch = stateBatch.cuda()
            rewardBatch = rewardBatch.cuda()
            nextStateBatch = nextStateBatch.cuda()

        qValues = model(stateBatch)
        model.eval()

        with torch.no_grad():
            nextPredictionBatch = model(nextStateBatch)

        model.train()

        yBatch = torch.cat(
            tuple(reward if done else reward + options.gamma * prediction for reward, done, prediction in
                  zip(rewardBatch, doneBatch, nextPredictionBatch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(qValues, yBatch)
        loss.backward()
        optimizer.step()

        print(
            f"""
                Epoch: {epoch}/{options.num_epochs}, Action: {action}, Score: {finalScore}, Tetrominoes {finalTetrominoes}, Cleared Lines: {finalClearedLines}
            """
        )

        writer.add_scalar("Train/Score", finalScore, epoch - 1)
        writer.add_scalar("Train/Tetrominoes", finalTetrominoes, epoch - 1)
        writer.add_scalar("Train/Cleared Lines", finalClearedLines, epoch - 1)

        os.makedirs(options.saved_path, exist_ok=True)

        if finalScore > bestScore:
            bestScore = finalScore
            torch.save(model, f"{options.saved_path}/tetris_best")

        if epoch > 0 and epoch % options.save_interval == 0:
            torch.save(model, f"{options.saved_path}/tetris_{epoch}")

    torch.save(model, f"{options.saved_path}/tetris")


def main() -> None:
    """
    Parse arguments and call train

    :return:
    """
    options = get_args()
    train(options)


if __name__ == "__main__":
    main()
