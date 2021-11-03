
import argparse
import torch
import cv2

from tetris import Tetris


def get_args():
    parser = argparse.ArgumentParser("Deep Q Network playing Tetris")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--render", default=False, help="Render the game", action=argparse.BooleanOptionalAction)
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="./trained_models/tetris")
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--verbose", default=False, help="Print score upon completion",
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--random_seed", type=int, default=0, help="Zero will use a random seed")

    args = parser.parse_args()
    return args


def test(options: argparse.Namespace) -> None:
    """
    Test by playing tetris

    :param options: argparse.Namespace instance holding testing parameters
    :return: None
    """
    totalScore = 0
    if torch.cuda.is_available():
        if options.random_seed == 0:
            torch.cuda.seed()
        else:
            torch.cuda.manual_seed(options.random_seed)
    else:
        if options.random_seed == 0:
            torch.seed()
        else:
            torch.manual_seed(options.random_seed)

    if torch.cuda.is_available():
        model = torch.load(options.saved_path)
    else:
        model = torch.load(options.saved_path, map_location=lambda storage, loc: storage)

    model.eval()
    env = Tetris(width=options.width, height=options.height, blockSize=options.block_size)
    env.reset()

    if torch.cuda.is_available():
        model.cuda()

    out = cv2.VideoWriter(options.output, cv2.VideoWriter_fourcc(*"MJPG"), options.fps,
                          (int(1.5 * options.width * options.block_size), options.height * options.block_size))

    while True:
        nextSteps = env.getNextStates()
        nextActions, nextStates = zip(*nextSteps.items())
        nextStates = torch.stack(nextStates)

        if torch.cuda.is_available():
            nextStates = nextStates.cuda()

        predictions = model(nextStates)[:, 0]
        index = torch.argmax(predictions).item()
        action = nextActions[index]

        score, done = env.step(action, render=options.render, video=out)

        totalScore += score

        if done:
            out.release()

            if options.verbose:
                print(f"""
                    Final score: {env.score}
                    Tetrominoes: {env.tetrominoes}
                    Cleared Lines: {env.clearedLines}
                    """
                )

            return env.score, env.tetrominoes, env.clearedLines


def main() -> None:
    """
    Parse arguments and test the saved network

    :return: None
    """
    options = get_args()
    test(options)


if __name__ == "__main__":
    main()
