import argparse

from learner import CartPoleLearner
from utils import set_seed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Run the training script",
                        action="store_true")
    parser.add_argument("--play", help="Run the inference script",
                        action='store_true')
    parser.add_argument("--resume", help="Resume from given savepoint",
                        action="store_true")
    parser.add_argument("--resume-episode",
                        type=str, default="latest")
    parser.add_argument("--data_path", help="Path to the folder for load and save",
                        type=str, default="save/")
    parser.add_argument("--device", help="Which device to use",
                        type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument("--seed", help="Random seed",
                        type=int, default=4242)
    parser.add_argument("--loss_fn", help="Which loss function to use",
                        type=str, default='L2', choices=['L1', 'L2'])
    parser.add_argument("--target-update-frequency", help="Target network synchronization frequency",
                        type=int, default=100)
    parser.add_argument("--max-frame", help="# of frames required to train",
                        type=int, default=1000000)
    args = parser.parse_args()
    return args


def main(args):
    set_seed(args.seed)
    learner = CartPoleLearner(
        device=args.device,
        loss_fn=args.loss_fn,
    )

    if args.train:
        if args.resume:
            learner.load(args.data_path, args.resume_episode)
        learner.train(
            target_update_frequency=args.target_update_frequency,
            max_frame=args.max_frame,
        )
    elif args.play:
        assert args.resume
        learner.load(args.data_path, args.resume_episode)
        learner.play()


if __name__ == "__main__":
    main(get_args())
