import argparse
import torch
import Learner
import datasets
import utils
from CGlowModel import CondGlowModel


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='predict c-Glow')

    # model parameters
    parser.add_argument("--x_size", type=tuple, default=(3,64,64))
    parser.add_argument("--y_size", type=tuple, default=(3,64,64))
    parser.add_argument("--x_hidden_channels", type=int, default=64)
    parser.add_argument("--x_hidden_size", type=int, default=128)
    parser.add_argument("--y_hidden_channels", type=int, default=256)
    parser.add_argument("-K", "--flow_depth", type=int, default=8)
    parser.add_argument("-L", "--num_levels", type=int, default=3)
    parser.add_argument("--learn_top", type=bool, default=True)

    # dataset
    parser.add_argument("-d", "--dataset_name", type=str, default="horse")
    parser.add_argument("-r", "--dataset_root", type=str, default="")
    parser.add_argument("--label_scale", type=float, default=1)
    parser.add_argument("--label_bias", type=float, default=0.5)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--x_bins", type=float, default=256.0)
    parser.add_argument("--y_bins", type=float, default=2.0)

    # output
    parser.add_argument("-o", "--out_root", type=str, default="")

    # model path
    parser.add_argument("--model_path", type=str, default="")

    # predictor parameters
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=10)


    args = parser.parse_args()
    cuda = torch.cuda.is_available()

    # dataset
    dataset = datasets.HorseDataset(args.dataset_root, (args.y_size[1], args.y_size[2]), args.y_size[0], "test")

    # model
    model = CondGlowModel(args)
    state = utils.load_state(args.model_path, cuda)
    model.load_state_dict(state["model"])
    del state

    # predictor
    predictor = Learner.Inferencer(model, dataset, args, cuda)

    # predict
    predictor.sampled_based_prediction(args.num_samples)

