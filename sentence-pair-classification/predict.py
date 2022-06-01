import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
warnings.simplefilter("ignore")

def get_probs_from_logits(logits):
    """
    Converts a tensor of logits into an array of probabilities by applying the sigmoid function
    """
    probs = torch.sigmoid(logits.unsqueeze(-1))
    return probs.detach().cpu().numpy()

def test_prediction(net, device, dataloader, with_labels=True, result_file="results/output.txt"):
    """
    Predict the probabilities on a dataset with or without labels and print the result in a file
    """
    net.eval()
    w = open(result_file, 'w')
    probs_all = []

    with torch.no_grad():
        if with_labels:
            for seq, attn_masks, token_type_ids, _ in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                logits = net(seq, attn_masks, token_type_ids)
                probs = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
                probs_all += probs.tolist()
        else:
            for seq, attn_masks, token_type_ids in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                logits = net(seq, attn_masks, token_type_ids)
                probs = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
                probs_all += probs.tolist()

    w.writelines(str(prob)+'\n' for prob in probs_all)
    w.close()

def validate_datafile(astring):
    if not astring.endswith('.tsv'):
        raise argparse.ArgumentTypeError("%s: is an invalid file, provide a tsv." % value)
    return astring

if __name__ == "__main__":

    print("Creation of the results' folder...")
    !mkdir results

    parser = ArgumentParser(description="Prediction with Fine-tuned ALBERT Sentence-Pair Classification Model", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_path", type=str, help="path to the fine-tuned model")
    parser.add_argument("test", type=validate_datafile, help="path to test dataset")

    args = parser.parse_args()

    path_to_model = args.model_path

    #path_to_model = '/content/models/albert-base-v2_lr_2e-05_val_loss_0.35007_ep_3.pt'

    path_to_output_file = 'results/output.txt'

    df_test = pd.read_csv(args.test)

    print("Reading test data...")
    test_set = CustomDataset(df_test, maxlen, bert_model)
    test_loader = DataLoader(test_set, batch_size=bs, num_workers=5)

    model = SentencePairClassifier(bert_model)
    if torch.cuda.device_count() > 1:  # if multiple GPUs
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    print()
    print("Loading the weights of the model...")
    model.load_state_dict(torch.load(path_to_model))
    model.to(device)

    print("Predicting on test data...")
    test_prediction(net=model, device=device, dataloader=test_loader, with_labels=True,  # set the with_labels parameter to False if your want to get predictions on a dataset without labels
                    result_file=path_to_output_file)
    print()
    print("Predictions are available in : {}".format(path_to_output_file))