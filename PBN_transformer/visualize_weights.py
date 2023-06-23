import argparse
import torch
from models import ProtoTEx
from IPython import embed
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_checkpoint",
        default="model_checkpoint.ckpt",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    model = torch.load(args.model_checkpoint)
        
    # embed()
    weight = torch.flatten(model["classfn_model.weight"]).cpu().detach().numpy()
    
    #create histogram with density curve overlaid
    # embed()
    sns.displot(weight, kde=True, bins=15)
    plt.savefig('hist_cosine.png')
    plt.show()