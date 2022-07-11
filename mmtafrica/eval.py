from datasets import load_dataset
import warnings
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.optimization import Adafactor
from transformers import get_linear_schedule_with_warmup
from tqdm.notebook import tqdm
import random
import sacrebleu
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
from joblib import Parallel, delayed,parallel_backend
import sys
from functools import partial
import json 
import time
import numpy as np
from datetime import datetime


class Config():
    def __init__(self,args) -> None:
        
        self.homepath = args.homepath
        self.prediction_path = os.path.join(args.homepath,args.prediction_path)
        # Use 'google/mt5-small' for non-pro cloab users
        self.model_repo = 'google/mt5-large'
        self.model_path_dir = args.homepath
        self.model_name = f'{args.model_name}.pt'
        self.bt_data_dir = os.path.join(args.homepath,args.bt_data_dir)

        #Data part
        self.parallel_dir= os.path.join(args.homepath,args.parallel_dir)
        self.mono_dir= os.path.join(args.homepath,args.mono_dir)

        #self.log = os.path.join(args.homepath,args.log)
        self.log = args.log
        self.mono_data_limit = args.mono_data_limit
        self.mono_data_for_noise_limit=args.mono_data_for_noise_limit
        #Training params
        self.n_epochs = args.n_epochs
        self.n_bt_epochs=args.n_bt_epochs

        self.batch_size = args.batch_size
        self.max_seq_len = args.max_seq_len
        self.min_seq_len = args.min_seq_len
        self.checkpoint_freq = args.checkpoint_freq
        self.lr = 1e-3
        self.print_freq = args.print_freq
        self.use_multiprocessing  = args.use_multiprocessing

        self.num_cores = mp.cpu_count() 
        self.NUM_PRETRAIN = args.num_pretrain_steps
        self.NUM_BACKTRANSLATION_TIMES =args.num_backtranslation_steps
        self.do_backtranslation=args.do_backtranslation
        self.now_on_bt=False
        self.bt_time=0
        self.using_reconstruction= args.use_reconstruction
        self.num_return_sequences_bt=2
        self.use_torch_data_parallel = args.use_torch_data_parallel

        self.gradient_accumulation_batch = args.gradient_accumulation_batch
        self.num_beams = args.num_beams

        self.best_loss = 1000
        self.best_loss_delta = 0.00000001
        self.patience=args.patience
        self.L2=0.0000001
        self.dropout=args.dropout

        self.drop_prob=args.drop_probability
        self.num_swaps=args.num_swaps

        self.verbose=args.verbose

        self.now_on_test=False

        #Initialization of state dict which will be saved during training
        self.state_dict = {'batch_idx': 0,'epoch':0,'bt_time':self.bt_time,'best_loss':self.best_loss}
        self.state_dict_check = {'batch_idx': 0,'epoch':0,'bt_time':self.bt_time,'best_loss':self.best_loss} #this is for tracing training after abrupt end!



        self.device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
            
        #We will be leveraging parallel and monolingual data for each of these languages.
        #parallel data will be saved in a central 'parallel_data 'folder as 'src'_'tg'_parallel.tsv 
        #monolingual data will be saved in another folder called 'monolingual_data' as 'lg'_mono.tsv

        #Each tsv file is of the form "input", "output"
        self.LANG_TOKEN_MAPPING = {
            'eng':'<eng>',
            'fon': '<fon>',
            'hau': '<hau>',
            'ibo': '<ibo>',
            'lug':'<lug>',
            'yor':'<yor>',
            'tsn':'<tsn>',
            'swh':'<swh>',
            'zul':'<zul>',
            'fra':'<fra>',
            'wol':'<wol>'
        }

        self.LANG_DICT = {'eng': ['hau', 'ibo', 'lug', 'swh', 'tsn', 'yor', 'zul'],
                          'fra':['wol']
                                }

        self.LANG_MAPPING = [f"{src}-{tgt}" for src in list(self.LANG_DICT.keys()) for tgt in self.LANG_DICT[src]]
        self.truncation=True


def beautify_time(time):
    hr = time//(3600)
    mins = (time-(hr*3600))//60
    rest = time -(hr*3600) - (mins*60)
    #DARIA's implementation!
    sp = ""
    if hr >=1:
        sp += '{} hours'.format(hr)
    if mins >=1:
        sp += ' {} mins'.format(mins)
    if rest >=1:
        sp += ' {} seconds'.format(rest)
    return sp


  
    

def get_dict(input,target,src,tgt):
  inp = [i for i in input]
  target_ = [ i for i in target]
  s= [src for i in range(len(inp))]
  t = [tgt for i in range(len(target_))]
  return [{'inputs':inp_,'targets':target__,'src':s_,'tgt':t_} for inp_,target__,s_,t_ in zip(inp,target_,s,t)]

def get_dict_mono(input,src,config):
  index = [i for i in range(len(input))]
  ids = random.sample(index,config.mono_data_limit)
  inp = [input[i] for i in ids]
  s= [src for i in range(len(inp))]
  data=[]
  for lang in config.LANG_TOKEN_MAPPING.keys():
    if f"{src}-{lang}" in config.LANG_MAPPING:
      data.extend([{'inputs':inp_,'src':s_,'tgt':lang} for inp_,s_ in zip(inp,s)])
  return data



def compress(input,target,src,tgt):
  return {'inputs':input,'targets':target,'src':src,'tgt':tgt}


def make_dataset(config,mode):
  if mode!='dev' and mode!='train' and mode!='test':
    raise Exception('mode is either train or dev or test!')
  else:
  
    files = [f.name for f in os.scandir(config.parallel_dir) ]
    files = [f for f in files if f.split('.')[-1]=='tsv' and f.split('.tsv')[0].endswith(mode) and len(f.split('_'))>2 ]
    data = [(f_.split('_')[0],f_.split('_')[1],pd.read_csv(os.path.join(config.parallel_dir,f_), sep="\t"))  for f_ in files]
    dict_ = [get_dict(df['input'],df['target'],src,tgt) for src,tgt,df in data if f"{src}-{tgt}" in config.LANG_MAPPING]
  return [item for sublist in dict_ for item in sublist]
  


def get_model_translation(config,model,tokenizer,sentence,tgt):
  if config.use_torch_data_parallel:
    max_seq_len_ = model.module.config.max_length
  else:
    max_seq_len_ = model.config.max_length
  input_ids = encode_input_str(config,text = sentence,target_lang = tgt,tokenizer = tokenizer,seq_len = max_seq_len_).unsqueeze(0).to(config.device)
  if config.use_torch_data_parallel:
    out = model.module.generate(input_ids,num_beams=3,do_sample=True, num_return_sequences=config.num_return_sequences_bt,max_length=config.max_seq_len,min_length=config.min_seq_len)
  else:
    out = model.generate(input_ids,num_beams=3, do_sample=True,num_return_sequences=config.num_return_sequences_bt,max_length=config.max_seq_len,min_length=config.min_seq_len)
    
  out_id = [i for i in range(config.num_return_sequences_bt)]
  id_ = random.sample(out_id,1)
  
  return tokenizer.decode(out[id_][0], skip_special_tokens=True)
  

def do_job(t,id_,tokenizers):
    tokenizer = tokenizers[id_ % len(tokenizers)]
    #We flip the input as target and vice versa in order to have target-side backtranslation (where source side is synthetic). 
    return {'inputs':get_model_translation(config,model,tokenizer,t['inputs'],t['tgt']),'targets':t['inputs'],'src':t['tgt'],'tgt':t['src']}


def do_job_pmap(t):
    #tokenizer = tokenizers[id_ % len(tokenizers)]
    return {'inputs':t['inputs'],'targets':get_model_translation(config,model,tokenizer,t['inputs'],t['tgt']),'src':t['src'],'tgt':t['tgt']}

def do_job_pool(bt_data,model,id_,tokenizers,config,mono_data):
    tokenizer = tokenizers[id_]
    if config.verbose:
        print(f"Mono data inside job pool: {mono_data}")
    sys.stdout.flush()
    res = [{'inputs':t['inputs'],'targets':get_model_translation(config,model,tokenizer,t['inputs'],t['tgt']),'src':t['src'],'tgt':t['tgt']} for t in mono_data]
    bt_data.put(res)
    return None
    

  

  

def encode_input_str(config,text, target_lang, tokenizer, seq_len):
  
  target_lang_token = config.LANG_TOKEN_MAPPING[target_lang]

  # Tokenize and add special tokens
  input_ids = tokenizer.encode(
      text = str(target_lang_token) + str(text),
      return_tensors = 'pt',
      padding = 'max_length',
      truncation = config.truncation,
      max_length = seq_len)

  return input_ids[0]
  
def encode_target_str(config,text, tokenizer, seq_len):
  token_ids = tokenizer.encode(
      text = str(text),
      return_tensors = 'pt',
      padding = 'max_length',
      truncation = config.truncation,
      max_length = seq_len)
  
  return token_ids[0]

def format_translation_data(config,sample,tokenizer,seq_len):
  
  # sample is of the form  {'inputs':input,'targets':target,'src':src,'tgt':tgt}

  # Get the translations for the batch

  input_lang = sample['src']
  target_lang = sample['tgt']


  input_text = sample['inputs']
  target_text = sample['targets']

  if input_text is None or target_text is None or f"{input_lang}-{target_lang}" not in config.LANG_MAPPING:
    return None

  input_token_ids = encode_input_str(config,input_text, target_lang, tokenizer, seq_len)
  
  target_token_ids = encode_target_str(config,target_text, tokenizer, seq_len)

  return input_token_ids, target_token_ids

def transform_batch(config,batch,tokenizer,max_seq_len):
  inputs = []
  targets = []
  for sample in batch:
    formatted_data = format_translation_data(config,sample,tokenizer,max_seq_len)
    
    if formatted_data is None:
      continue
    
    input_ids, target_ids = formatted_data
    inputs.append(input_ids.unsqueeze(0))
    targets.append(target_ids.unsqueeze(0))
    
  if inputs==[] and targets==[]:
    return None,None

  batch_input_ids = torch.cat(inputs)
  batch_target_ids = torch.cat(targets)

  return batch_input_ids, batch_target_ids

def get_data_generator(config,dataset,tokenizer,max_seq_len,batch_size):
  random.shuffle(dataset)
 
  for i in range(0, len(dataset), batch_size):
    raw_batch = dataset[i:i+batch_size]
    yield transform_batch(config,raw_batch, tokenizer,max_seq_len)

def eval_model(config,tokenizer,model, gdataset, max_iters=8):
  test_generator = get_data_generator(config,gdataset,tokenizer,config.max_seq_len, config.batch_size)
  eval_losses = []
  for i, (input_batch, label_batch) in enumerate(test_generator):
 
    input_batch, label_batch = input_batch.to(config.device), label_batch.to(config.device)
    model_out = model.forward(
        input_ids = input_batch,
        labels = label_batch)
    
    if config.use_torch_data_parallel:
        loss = torch.mean(model_out.loss)
    else:
        loss = model_out.loss    
    
    eval_losses.append(loss.item())

  return np.mean(eval_losses)



def evaluate(config,tokenizer,model,test_dataset,src_lang=None,tgt_lang=None):
  if src_lang!=None and tgt_lang!=None and f"{src_lang}-{tgt_lang}" in config.LANG_MAPPING:
    if config.verbose:
        with open(config.log,'a+') as fl:
          print(f"Getting evaluation set for source language -> {src_lang} and target language -> {tgt_lang}",file=fl)
    data = [t for t in test_dataset if t['src']==src_lang and t['tgt']==tgt_lang]
    
  else:
    data= [t for t in test_dataset]
    
  inp = [t['inputs'] for t in data]
  truth = [t['targets'] for t in data]
  tgt_lang_ = [t['tgt'] for t in data]

  seq_len__ = config.max_seq_len
 
  input_tokens = [encode_input_str(config,text = inp[i],target_lang = tgt_lang_[i],tokenizer = tokenizer,seq_len =seq_len__).unsqueeze(0).to(config.device) for i in range(len(inp))]
  
  if config.use_torch_data_parallel:
    output = [model.module.generate(input_ids, num_beams=config.num_beams, num_return_sequences=1,max_length=config.max_seq_len,min_length=config.min_seq_len) for input_ids in tqdm(input_tokens)]
  else:
    output = [model.generate(input_ids, num_beams=config.num_beams, num_return_sequences=1,max_length=config.max_seq_len,min_length=config.min_seq_len) for input_ids in tqdm(input_tokens)]
  output = [tokenizer.decode(out[0], skip_special_tokens=True) for out in tqdm(output)]
  
  df= pd.DataFrame({'predictions':output,'truth':truth,'inputs':inp})
  if config.now_on_bt and config.using_reconstruction:
    filename = f'{src_lang}_{tgt_lang}_bt_{config.bt_time}_rec.tsv'
  elif config.now_on_bt:
    filename = f'{src_lang}_{tgt_lang}_bt_{config.bt_time}.tsv'
  elif config.now_on_test:
    filename = f'{src_lang}_{tgt_lang}_TEST.tsv'
  else: 
    filename = f'{src_lang}_{tgt_lang}.tsv'
  df.to_csv(os.path.join(config.prediction_path,filename),sep='\t',index=False)
  try:
    spbleu = sacrebleu.corpus_bleu(output, [truth])
  except Exception:
    warnings.warn(f'There is a problem with {src_lang}_{tgt_lang}. Truth is {truth} \n Input is {inp} \n Setting bleu for this to 0 ')
    return 0 
 
  return spbleu.score


def do_evaluation(config,tokenizer,model,test_dataset):
    LANGS = list(config.LANG_TOKEN_MAPPING.keys())
    if config.now_on_bt and config.using_reconstruction:
        s=f'---------------------------AFTER BACKTRANSLATION {config.bt_time} with RECONSTRUCTION---------------------------'+'\n'
    elif config.now_on_bt:
         s=f'---------------------------AFTER BACKTRANSLATION {config.bt_time}---------------------------'+'\n'
    elif config.now_on_test:
        s=f'---------------------------TESTING EVALUATION---------------------------'+'\n'
    else:
        s=f'---------------------------EVALUATION ON DEV---------------------------'+'\n'
    for i in range(len(LANGS)):
        for j in range(len(LANGS)):
            if f"{LANGS[i]}-{LANGS[j]}" in config.LANG_MAPPING:
                eval_bleu = evaluate(config,tokenizer,model,test_dataset,src_lang=LANGS[i],tgt_lang=LANGS[j])
                a = f'Bleu Score for {LANGS[i]} to {LANGS[j]} -> {eval_bleu} '+'\n'
                s+=a
    s+='------------------------------------------------------'     
    with open(os.path.join(config.homepath,'bleu_log.txt'), 'a+') as fl:
        print(s,file=fl)
    

def main(args):
    if not os.path.exists(args.homepath):
        os.makedirs(args.homepath,exist_ok=True)
        #raise Exception(f'HOMEPATH {args.homepath} does not exist!')
    config = Config(args)
    if not os.path.exists(config.prediction_path):
        os.makedirs(config.prediction_path)
    if not os.path.exists(config.bt_data_dir):
        os.makedirs(config.bt_data_dir)
    """# Load Tokenizer & Model"""

    tokenizer = AutoTokenizer.from_pretrained(config.model_repo)
    if config.use_multiprocessing:
        tokenizers_for_parallel = [AutoTokenizer.from_pretrained(config.model_repo) for i in range(config.num_cores)]

    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_repo)

    if not os.path.exists(config.parallel_dir):
        raise Exception(f'Directory `{config.parallel_dir}` cannot be empty! It must contain the parallel files')
    
    dev_dataset = make_dataset(config,'dev')
    with open(config.log,'a+') as fl:
        print(f"Length of dev dataset: {len(dev_dataset)}",file=fl)

    """## Update tokenizer"""
    special_tokens_dict = {'additional_special_tokens': list(config.LANG_TOKEN_MAPPING.values())}
    tokenizer.add_special_tokens(special_tokens_dict)
    if config.use_multiprocessing:
        for tk in tokenizers_for_parallel:
            tk.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))


    """# Train/Finetune MT5"""
    if not os.path.exists(args.checkpoint) or args.checkpoint is None:
      raise Exception(f"Path to checkpoint does not exist: `{args.checkpoint}`") 
    if os.path.exists(args.checkpoint):
        if config.verbose:
            with open(config.log,'a+') as fl:
                print(f"-----------Using model checkpoint for evaluation => {args.checkpoint}-----------",file=fl)
        
        state_dict = torch.load(args.checkpoint)
          
    
        # Note to self: Make this beter.
        config.state_dict_check['epoch']=state_dict['epoch']
        config.state_dict_check['bt_time']=state_dict['bt_time']
        config.state_dict_check['best_loss']=state_dict['best_loss']
        config.best_loss = config.state_dict_check['best_loss']
        config.state_dict_check['batch_idx']=state_dict['batch_idx']
        model.load_state_dict(state_dict['model_state_dict'])
      
        #Temp change
        config.state_dict_check['epoch']=-1
        config.state_dict_check['batch_idx']=0
        config.state_dict_check['bt_time']=-1
      

    #Using DataParallel
    if config.use_torch_data_parallel:
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model = model.to(config.device)
    #-----

    # Optimizer
    optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=config.lr)

  
    if config.verbose:
        with open(config.log,'a+') as fl:
            print('Evaluaton...',file=fl)
    do_evaluation(config,tokenizer,model,dev_dataset)

    with open(config.log,'a+') as fl:
        print('Evaluating on test set',file=fl)
    test_dataset = make_dataset(config,'test')
    with open(config.log,'a+') as fl:
        print(f"Length of test dataset: {len(test_dataset)}",file=fl)
    do_evaluation(config,tokenizer,model,test_dataset)

    with open(config.log,'a+') as fl:
        print("ALL DONE",file=fl)

if __name__=="__main__":
    from argparse import ArgumentParser
    import json
    import os
    
    
    parser = ArgumentParser('MMTArica Experiments')

    parser.add_argument('--checkpoint', type=str, default=None,
        help="Path to checkpoint. (default: None)")

    parser.add_argument('--homepath', type=str, default=os.getcwd(),
        help="Homepath directory. Where all experiments are saved and all \
        necessary files/folders are saved. (default: current working directory)")
    
    parser.add_argument('--prediction_path', type=str, default='./predictions',
        help='directory path to save predictions (default: %(default)s)')
    
    parser.add_argument('--model_name', type=str, default='mmt_translation',
        help='Name of model (default: %(default)s)')

    parser.add_argument('--bt_data_dir', type=str, default='btData',
        help='Directory to save back-translation files (default: %(default)s)')   

    parser.add_argument('--parallel_dir', type=str, default='parallel',
        help='name of directory where parallel corpora is saved') 

    parser.add_argument('--mono_dir', type=str, default='mono',
        help='name of directory where monolingual files are saved (default: %(default)s)')                    
    
    parser.add_argument('--log', type=str, default='train.log',
        help='name of file to log experiments (default: %(default)s)')

    parser.add_argument('--mono_data_limit', type=int, default=300,
        help='limit of monolingual sentences to use for training (default: %(default)s)')

    parser.add_argument('--mono_data_for_noise_limit', type=int, default=50,
        help='limit of monolingual sentences to use for noise (default: %(default)s)')

    parser.add_argument('--n_epochs', type=int, default=10,
        help='number of training epochs (default: %(default)s)')            
    
    parser.add_argument('--n_bt_epochs', type=int, default=3,
        help='number of backtranslation epochs (default: %(default)s)')

    parser.add_argument('--batch_size', type=int, default=64,
        help='batch size (default: %(default)s)')

    parser.add_argument('--max_seq_len', type=int, default=50,
        help='maximum length of sentence. All sentences beyond this length will be skipped. (default: %(default)s)')

    parser.add_argument('--min_seq_len', type=int, default=2,
        help='mnimum length of sentence. All sentences beyond this length will be skipped. (default: %(default)s)')

    parser.add_argument('--checkpoint_freq', type=int, default=10_000,
    help='maximum length of sentence. All sentences beyond this length will be skipped. (default: %(default)s)')

    parser.add_argument('--lr', type=float, default=1e-4,
    help='learning rate. (default: %(default)s)')

    parser.add_argument('--print_freq', type=int, default=5_000,
    help='frequency at which to print to log. (default: %(default)s)') 

    parser.add_argument('--use_multiprocessing', type=bool, default=False,
    help='whether or not to use multiprocessing. (default: %(default)s)') 

    parser.add_argument('--num_pretrain_steps', type=int, default=20,
    help='number of pretrain steps. (default: %(default)s)') 

    parser.add_argument('--num_backtranslation_steps', type=int, default=5,
    help='number of pretrain steps. (default: %(default)s)') 

    parser.add_argument('--do_backtranslation', type=bool, default=False,
    help='whether or not to do backtranslation during training. (default: %(default)s)')

    parser.add_argument('--use_reconstruction', type=bool, default=False,
    help='whether or not to use reconstruction during training. (default: %(default)s)')

    parser.add_argument('--use_torch_data_parallel', type=bool, default=False,
    help='whether or not to use torch data parallelism. (default: %(default)s)')

    parser.add_argument('--gradient_accumulation_batch', type=int, default=4096//64,
    help='batch size for gradient accumulation. (default: %(default)s)')

    parser.add_argument('--num_beams', type=int, default=4,
    help='number of beams to use for inference. (default: %(default)s)')

    parser.add_argument('--patience', type=int, default=15_000_000,
    help='patience for early stopping. (default: %(default)s)')

    parser.add_argument('--drop_probability', type=float, default=0.2,
    help='drop probability for reconstruction. (default: %(default)s)')

    parser.add_argument('--dropout', type=float, default=0.1,
    help='dropout probability. (default: %(default)s)')

    parser.add_argument('--num_swaps', type=int, default=3,
    help='number of word swaps to perform during reconstruction. (default: %(default)s)')

    parser.add_argument('--verbose', type=bool, default=True,
    help='whether or not to print information during experiments. (default: %(default)s)')

    args = parser.parse_args() 
    

    main(args)    
    
    
