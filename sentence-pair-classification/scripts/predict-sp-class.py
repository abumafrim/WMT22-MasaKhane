# -*- coding: utf-8 -*-
"""Fine_tune_ALBERT_sentence_pair_classification.ipynb

## Installation of libraries and imports
"""

import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import psutil
import humanize
import os
import GPUtil as GPU

from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentTypeError
import warnings
warnings.simplefilter("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check that we are using 100% of GPU memory footprint support libraries/code
# from https://github.com/patrickvonplaten/notebooks/blob/master/PyTorch_Reformer.ipynb

os.system("ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi")

def printm(gpu):
    process = psutil.Process(os.getpid())
    print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))

GPUs = GPU.getGPUs()
# XXX: only one GPU on Colab and isn’t guaranteed
if len(GPUs) > 0:
    gpu = GPUs[0]
    printm(gpu)

"""## Classes and functions"""

class CustomDataset(Dataset):

    def __init__(self, data, maxlen, with_labels=True, bert_model='albert-base-v2'):

        self.data = data  # pandas dataframe
        #Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)  

        self.maxlen = maxlen
        self.with_labels = with_labels 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent1 = str(self.data.loc[index, 'sentence1'])
        sent2 = str(self.data.loc[index, 'sentence2'])

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent1, sent2, 
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  # Return torch.Tensor objects
        
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label = self.data.loc[index, 'label']
            return token_ids, attn_masks, token_type_ids, label  
        else:
            return token_ids, attn_masks, token_type_ids

class SentencePairClassifier(nn.Module):

    def __init__(self, bert_model="albert-base-v2", freeze_bert=False):
        super(SentencePairClassifier, self).__init__()
        #  Instantiating BERT-based model object
        self.bert_layer = AutoModel.from_pretrained(bert_model)

        #  Fix the hidden-state size of the encoder outputs (If you want to add other pre-trained models here, search for the encoder output size)
        if bert_model == "albert-base-v2":  # 12M parameters
            hidden_size = 768
        elif bert_model == "albert-large-v2":  # 18M parameters
            hidden_size = 1024
        elif bert_model == "albert-xlarge-v2":  # 60M parameters
            hidden_size = 2048
        elif bert_model == "albert-xxlarge-v2":  # 235M parameters
            hidden_size = 4096
        elif bert_model == "bert-base-uncased": # 110M parameters
            hidden_size = 768

        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(p=0.1)

    @autocast()  # run in mixed precision
    def forward(self, input_ids, attn_masks, token_type_ids):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''

        # Feeding the inputs to the BERT-based model to obtain contextualized representations
        cont_reps, pooler_output = self.bert_layer(input_ids, attn_masks, token_type_ids)

        # Feeding to the classifier layer the last layer hidden-state of the [CLS] token further processed by a
        # Linear Layer and a Tanh activation. The Linear layer weights were trained from the sentence order prediction (ALBERT) or next sentence prediction (BERT)
        # objective during pre-training.
        logits = self.cls_layer(self.dropout(pooler_output))

        return logits

def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_probs_from_logits(logits):
    """
    Converts a tensor of logits into an array of probabilities by applying the sigmoid function
    """
    probs = torch.sigmoid(logits.unsqueeze(-1))
    return probs.detach().cpu().numpy()

def test_prediction(net, device, dataloader, with_labels, result_file="results/output.txt"):
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
    raise ArgumentTypeError("%s: is an invalid file, provide a tsv file." % astring)
  return astring


if __name__ == "__main__":
    
  parser = ArgumentParser(description="Sentence-Pair Classification Prediction", formatter_class=ArgumentDefaultsHelpFormatter)
  
  parser.add_argument("--predict", action='store_true')
  parser.add_argument("--model", type=str, default='albert-base-v2', help="model to finetune: 'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2', 'bert-base-uncased', ...")
  parser.add_argument("--model_path", type=str, required=True,help="path to sp-class. model")
  parser.add_argument("--data_path", type=validate_datafile, help="path to sentence pairs to predict")
  parser.add_argument("--output_file", type=str, help="file to save predictions")
  parser.add_argument("-f", "--freeze_bert", default=False, help="if True, freeze the encoder weights and only update the classification layer weights")
  parser.add_argument("-l", "--maxlen", type=int, default=128, help="maximum length of the tokenized input sentence pair: if greater than 'maxlen', the input is truncated and else if smaller, the input is padded")
  parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size")
  parser.add_argument("-s", "--seed", type=int, default=1, help="seeds")
      
  args = parser.parse_args()

  if not os.path.exists(args.output_file):
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

  bert_model = args.model
  maxlen = args.maxlen
  bs = args.batch_size

  #  Set all seeds to make reproducible results
  set_seed(args.seed)

  path_to_output_file = args.output_file
  path_to_model = args.model_path

  df_pred = pd.read_csv(args.data_path, sep='\t')

  print("Reading data...")
  pred_set = CustomDataset(df_pred, maxlen, False, bert_model)
  pred_loader = DataLoader(pred_set, batch_size=bs, num_workers=5)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model = SentencePairClassifier(bert_model)
  if torch.cuda.device_count() > 1:  # if multiple GPUs
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      model = nn.DataParallel(model)

  print()
  print("Loading the weights of the model...")
  model.load_state_dict(torch.load(path_to_model))
  model.to(device)

  print("Predicting quality of parallel data...")
  test_prediction(net=model, device=device, dataloader=pred_loader, with_labels=False, result_file=path_to_output_file)
  # set the with_labels parameter to False if your want to get predictions on a dataset without labels

  print("\nPredictions are available in : {}".format(path_to_output_file))