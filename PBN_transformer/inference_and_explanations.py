# TODO: The whole code should be changed according to the
# new structure in the main and evaluate_model files


import torch
from transformers import AutoTokenizer
from preprocess import *
from IPython import embed
import os
import argparse
import utils
import joblib
from models import ProtoTEx
from models_electra import ProtoTEx_Electra
import sys

sys.path.append("../datasets")
import configs


def main(args):
    # preprocess the propaganda dataset loaded in the data folder. Original dataset can be found here
    # https://propaganda.math.unipd.it/fine-grained-propaganda-emnlp.html

    if args.architecture == "BART":
        tokenizer = AutoTokenizer.from_pretrained("ModelTC/bart-base-mnli")
        # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    elif args.architecture == "ELECTRA":
        tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
    else:
        print(f"Invalid backbone architecture: {args.architecture}")

    all_datasets = utils.load_dataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=configs.dataset_to_max_length[args.dataset],
    )

    all_dataloaders = {
        dataset_name: torch.utils.data.DataLoader(
            all_datasets[dataset_name],
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: {
                "input_ids": torch.LongTensor([i["input_ids"] for i in batch]),
                "attention_mask": torch.Tensor([i["attention_mask"] for i in batch]),
                "label": torch.LongTensor([i["label"] for i in batch]),
            },
        )
        for dataset_name in all_datasets.keys()
    }

    if args.model == "ProtoTEx":
        print("ProtoTEx best model: {0}".format(args.num_prototypes))
        if args.architecture == "BART":
            print(f"Using backone: {args.architecture}")
            torch.cuda.empty_cache()
            model = ProtoTEx(
                num_prototypes=args.num_prototypes,
                class_weights=None,
                n_classes=configs.dataset_to_num_labels[args.dataset],
                max_length=configs.dataset_to_max_length[args.dataset],
                bias=False,
                special_classfn=True,
                p=1,  # p=0.75,
                batchnormlp1=True,
            )
        elif args.architecture == "ELECTRA":
            model = ProtoTEx_Electra(
                num_prototypes=args.num_prototypes,
                class_weights=None,
                n_classes=configs.dataset_to_num_labels[args.dataset],
                max_length=configs.dataset_to_max_length[args.dataset],
                bias=False,
                special_classfn=True,
                p=1,  # p=0.75,
                batchnormlp1=True,
            )

        else:
            print(f"Invalid backbone architecture: {args.architecture}")

    print(f"Loading model checkpoint: Models/{args.modelname}")
    pretrained_dict = torch.load(f"Models/{args.modelname}")
    # Fiter out unneccessary keys
    model_dict = model.state_dict()
    filtered_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_dict[k] = v
        else:
            print(f"Skipping weights for: {k}")
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = torch.nn.DataParallel(model)
    model = model.to(device)

    if not os.path.exists(f"artifacts/{args.dataset}"):
        os.mkdir(f"artifacts/{args.dataset}")
    if not os.path.exists(f"artifacts/{args.dataset}/{args.modelname}"):
        os.mkdir(f"artifacts/{args.dataset}/{args.modelname}")

    bestk_train_data_per_proto = utils.get_bestk_train_data_for_every_proto(
        all_dataloaders["test"], model_new=model, top_k=5
    )
    joblib.dump(
        bestk_train_data_per_proto,
        f"artifacts/{args.dataset}/{args.modelname}/bestk_train_data_per_proto.joblib",
    )

    best_protos_per_traineg = utils.get_best_k_protos_for_batch(
        dataloader=all_dataloaders["test"],
        model_new=model,
        topk=5,
        do_all=True,
    )
    joblib.dump(
        best_protos_per_traineg,
        f"artifacts/{args.dataset}/{args.modelname}/best_protos_per_traineg.joblib",
    )

    best_protos_per_split = {}
    for dataset_name, dataloader in all_dataloaders.items():
        if not (dataset_name.startswith("test_") or dataset_name.startswith("adv_")):
            continue
        print(f"getting the prototypes for test examples of {dataset_name}")

        best_protos_per_split[dataset_name] = utils.get_best_k_protos_for_batch(
            dataloader=dataloader,
            model_new=model,
            topk=5,
            do_all=True,
        )

    joblib.dump(
        best_protos_per_split,
        f"artifacts/{args.dataset}/{args.modelname}/best_protos_per_testeg.joblib",
    )
    torch.save(
        model.prototypes.detach().cpu().numpy(),
        f"artifacts/{args.dataset}/{args.modelname}/all_protos.pt",
    )

    # train_sents_joined = train_sentences
    # test_sents_joined = test_sentences

    """
    distances generation
    test true labels and pred labels
    """
    # loader = tqdm(test_dl, total=len(test_dl), unit="batches")
    # model.eval()
    # with torch.no_grad():
    #     test_true=[]
    #     test_pred=[]
    #     for batch in loader:
    #         input_ids,attn_mask,y=batch
    #         classfn_out,_=model(input_ids,attn_mask,y,use_decoder=False,use_classfn=1)
    #         predict=torch.argmax(classfn_out,dim=1)
    # #         correct_idxs.append(torch.nonzero((predicted==y.cuda())).view(-1)
    #         test_pred.append(predict.cpu().numpy())
    #         test_true.append(y.cpu().numpy())
    # test_true=np.concatenate(test_true)
    # test_pred=np.concatenate(test_pred)

    """
    distances generation
    csv generation
    """
    # import csv

    # fields = ["S.No.", "Test Sentence","Predicted","Actual","Actual Prop or NonProp"]
    # num_protos_per_test=5
    # num_train_per_proto=5
    # for i in range(num_protos_per_test):
    #     for j in range(num_train_per_proto):
    #         fields.append(f"Prototype_{i}_wieght0")
    #         fields.append(f"Prototype_{i}_wieght1")
    #         fields.append(f"Prototype_{i}_Nearest_train_eg_{j}")
    #         fields.append(f"Prototype_{i}_Nearest_train_eg_{j}_actuallabel")
    #         fields.append(f"Prototype_{i}_Nearest_train_eg_{j}_distance")

    # filename = f"{model_path[len('Models/'):]}_nearest.csv"
    # weights=model.classfn_model.weight.detach().cpu().numpy()
    # with open(filename, 'w') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(fields)
    #     for i in range(len(test_sents_joined)):
    # #     for i in range(100):
    #         row=[i,test_sents_joined[i],test_pred[i],test_labels[i],test_true[i]]
    #         for j in range(num_protos_per_test):
    #             proto_idx=best_protos_per_testeg[0][i][j]
    #             for k in range(num_train_per_proto):
    # #                 print(i,j,k)
    #                 row.append(weights[0][proto_idx])
    #                 row.append(weights[1][proto_idx])
    #                 row.append(train_sents_joined[bestk_train_data_per_proto[0][proto_idx][k]])
    #                 row.append(train_labels[bestk_train_data_per_proto[0][proto_idx][k]])
    #                 row.append(bestk_train_data_per_proto[1][k][proto_idx])

    #         csvwriter.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tiny_sample", dest="tiny_sample", action="store_true")
    # parser.add_argument("--nli_dataset", help="check if the dataset is in nli
    # format that has sentence1, sentence2, label", action="store_true")
    parser.add_argument("--num_prototypes", type=int, default=16)
    parser.add_argument("--model", type=str, default="ProtoTEx")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--modelname", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--learning_rate", type=float, default="3e-5")

    # Wandb parameters
    parser.add_argument("--project", type=str)
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--nli_intialization", type=str, default="Yes")
    parser.add_argument("--none_class", type=str, default="No")
    parser.add_argument("--curriculum", type=str, default="No")
    parser.add_argument("--augmentation", type=str, default="No")
    parser.add_argument("--architecture", type=str, default="BART")

    args = parser.parse_args()

    main(args)
