# coding=utf-8

"""
Vocabulary helper class
"""

import re
import numpy as np
import json
import os
import torch


def convert_json_key(param_dict):
    """
    json.dump不支持key是int的dict，在编码存储的时候会把所有的int型key写成str类型的
    所以在读取json文件后，用本方法将所有的被解码成str的int型key还原成int
    """
    new_dict = dict()
    for key, value in param_dict.items():
        if isinstance(value, (dict,)):
            res_dict = convert_json_key(value)
            try:
                new_key = int(key)
                new_dict[new_key] = res_dict
            except:
                new_dict[key] = res_dict
        else:
            try:
                new_key = int(key)
                new_dict[new_key] = value
            except:
                new_dict[key] = value

    return new_dict



def load_json_file(file_path):
    with open(file_path, "r", encoding="utf8") as f:
        # return convert_json_key(json.load(f))
        return json.load(f)
        
def save_json(save_path, data):
    assert save_path.split('.')[-1] == 'json'
    with open(save_path,'w') as file:
        json.dump(data, file)



# contains the data structure
class Vocabulary:
    """Stores the tokens and their conversion to vocabulary indexes."""

    def __init__(self, tokens=None, starting_id=0, loading=True, vocab_path=None, updating=False):

        self.loading = loading
        self.updating = updating
        self.vocab_path = 'configs/vocab.json' if not vocab_path else vocab_path
        self._tokens = load_json_file(self.vocab_path) if self.loading else {} 
        self.decode_tokens = {v:k for k,v in self._tokens.items()}
        self._current_id = starting_id if not loading else len(self._tokens)

        if tokens:
            for token, idx in tokens.items():
                self._add(token, idx)
                self._current_id = max(self._current_id, idx + 1)

    def __getitem__(self, token_or_id):
        return self._tokens[token_or_id]

    def add(self, token):
        """Adds a token."""
        if not isinstance(token, str):
            raise TypeError("Token is not a string")
        if token in self:
            return self[token]
        self._add(token, self._current_id)
        self._current_id += 1
        return self._current_id - 1

    def update(self, tokens):
        """Adds many tokens."""
        return [self.add(token) for token in tokens]

    def __delitem__(self, token_or_id):
        other_val = self._tokens[token_or_id]
        del self._tokens[other_val]
        del self._tokens[token_or_id]

    def __contains__(self, token_or_id):
        return token_or_id in self._tokens

    def __eq__(self, other_vocabulary):
        return self._tokens == other_vocabulary._tokens  # pylint: disable=W0212

    def __len__(self):
        return len(self._tokens)

    def encode(self, tokens):
        """Encodes a list of tokens as vocabulary indexes."""
        vocab_index = np.zeros(len(tokens), dtype=np.float32)
        for i, token in enumerate(tokens):
            try:
                vocab_index[i] = self._tokens[token]
            except:
                save_fn = 'configs/tranning_updating_vocab.json'
                print(f"Find unknow token {token} in vocabulary, setting to 'UNK' and add to vocab, save to {save_fn}")
                vocab_index[i] = self._tokens["<PAD>"]
                if self.updating: # updating vocab and save
                    self.update([token])
                    self.save(save_fn)

        return vocab_index

    def decode(self, vocab_index):
        """Decodes a vocabulary index matrix to a list of tokens."""
        tokens = []
        for idx in vocab_index:
            tokens.append(self.decode_tokens[idx])
        return tokens

    def _add(self, token, idx):
        if idx not in self._tokens:
            self._tokens[token] = idx
            # self._tokens[idx] = token
        else:
            raise ValueError("IDX already present in vocabulary")

    def tokens(self):
        """Returns the tokens from the vocabulary"""
        return self._tokens
        # return [t for t in self._tokens if isinstance(t, str)]

    def save(self, path):
        """
        save vocab to file
        """
        save_json(path, self._tokens)





# 分词器，对aa，cds以及utr采用不同的分词器
        
class ntTokenizer():
    """
    核酸分词方法

    """

    def __init__(self, tokenizer_fn=None):
        """
        
        """
        # self.vocab = load_json_file(tokenizer_fn) if tokenizer_fn is not None else 
        self.vocab = self._get_vocab()

    def _get_vocab(self):
        return {w: idx for idx, w in enumerate(['A', 'T', 'C', 'G', 'N'])}

    def tokenize(self, sentence, truncation=True, padding=True, ids=False, token="byte", with_begin_and_end=True):
        """
        generate tokens based tokenizer
        
        sentence: str, 
        token: 分词方法，word or byte
        """

        vocab = self._get_vocab()
        tokens = list(sentence.lower().strip())

        if with_begin_and_end:
            if token == 'utr5':
                tokens = ['<UTR5_ST>'] + tokens + ['<UTR5_ED>']

            elif token == 'utr3':
                tokens = ['<UTR3_ST>'] + tokens + ['<UTR3_ED>']

            else:
                tokens = ['<ST>'] + tokens + ['<ED>']

        if ids:
            tokens = [vocab[i] for i in tokens]
 
        

        return tokens

    def decode(self, ids):
        """
        decode from vocabulary indices to raw sentence string.
        """
        vocab_reverse = {v:k for k,v in self._get_vocab().items()}
        decoded_string = ''.join([vocab_reverse[id] for id in ids])
        
        return decoded_string


        
class CDSTokenizer():
    """
    基于规则的分词器
    
    对CDS核酸序列与AA序列，采用基于规则的分词策略
    1）对CDS：每三个碱基作为一个词（密码子）
    2）对氨基酸：每一个氨基酸单独作为一个词
    """
    
    
    def __init__(self, codon_vocab_fn=None, aa_vocab_fn=None):
        """
        
        """
        
        self.config_dir = os.path.join(os.path.dirname(__file__), '../configs')
        
        _, self.codon_vocab = self.__load_list_file(codon_vocab_fn if codon_vocab_fn else "codon_vocab.json",
                                               offset=0, verbose=False)
        _, self.aa_vocab = self.__load_list_file(aa_vocab_fn if aa_vocab_fn else "aa_vocab.json", 
                                            offset=0, verbose=False)
    
    def _get_vocab(self, added_only=False):
        """
        词表
        
        return:
            密码子词表
            氨基酸词表
        """
        
        return self.codon_vocab, self.aa_vocab
    

    def __load_list_file(self, file_name, offset=0, verbose=False):
        """
        加载
        """
        file_path = os.path.join(self.config_dir, file_name)
        if not os.path.exists(file_path):
            raise ValueError('"{}" file does not exist.'.format(file_path))
        else:
            elements = load_json_file(file_path)
            elements_dict = {w: idx + offset for idx, w in enumerate(elements)}
            if verbose:
                print("config {} loaded".format(file_path))
            return elements, elements_dict
        
        
    def tokenize(self, sentence, truncation=True, padding=True, ids=False, token="cds", with_begin_and_end=True):
        """
        分词
        
        params:
            token: 指定分词类型，支持cds与aa两类
            ids: 是否返回ids
        """
        
        sentence = sentence.upper().strip().replace(' ', '')
        if  token=="cds":
            tokens = [sentence[i:i+3] for i in range(0, len(sentence), 3)]
            if with_begin_and_end:
                tokens = ["<CDS_ST>"] + tokens + ["<CDS_ED>"]
            if ids:
                return list(itemgetter(*tokens)(self.codon_vocab))
            return tokens
        
        if  token=="aa":
            tokens = list(sentence)
            if with_begin_and_end:
                tokens = ["<AA_ST>"] + tokens + ["<AA_ED>"]
            if ids:
                return list(itemgetter(*tokens)(self.aa_vocab))
            return tokens
        


def create_vocabulary(smiles_list, tokenizer):
    """Creates a vocabulary for the SMILES syntax."""
    tokens = set()
    for smi in smiles_list:
        tokens.update(tokenizer.tokenize(smi, with_begin_and_end=False))

    vocabulary = Vocabulary()
    vocabulary.update(["<pad>", "$", "^"] + sorted(tokens))
    return vocabulary



class Tokenizer():
    """
    词编码器
    """

    def __init__(self, seq_type='aa'):

        self.seq_type = seq_type
        self.vocabulary = Vocabulary(loading=True)
        self.nttokenizer = ntTokenizer()
        self.cdsTokenizer = CDSTokenizer()

    def encode(self, tokens):
        """
        encode a list of tokens to a series of numbers
        """
        return self.vocabulary.encode(tokens).astype(int).tolist()

    def decode(self, ids):
        return self.vocabulary.decode(ids)

    def tokenize(self, tokens, seq_type=None):
        """
        split strings to tokens
        """
        
        seq_type = seq_type if seq_type else self.seq_type
        if seq_type == 'aa':
            prompt_token = self.cdsTokenizer.tokenize(tokens, token="aa")
        elif seq_type == 'cds':
            prompt_token = self.cdsTokenizer.tokenize(tokens, token="cds")
        
        elif  seq_type == 'utr5':
            prompt_token = self.nttokenizer.tokenize(tokens, token='utr5')
            
        elif seq_type == 'utr3':
            prompt_token = self.nttokenizer.tokenize(tokens, token='utr3')
        else:
            prompt_token = list(tokens)
        return prompt_token


    def tokenize_to_ids(self, tokens, seq_type=None):
        """
        encode a strings to number
        """
        seq_type = seq_type if seq_type else self.seq_type

        return self.encode(self.tokenize(tokens, seq_type))


def get_vocab():
    """
    获得汇总词表
    
    """
        
    
    _inverse_trans_vocab_vocab_fn = os.path.join(os.path.dirname(__file__), 
                                                 '../configs/inverse_trans_vocab.json')
    if os.path.exists(_inverse_trans_vocab_vocab_fn):
        inverse_trans_vocab = load_json_file(_inverse_trans_vocab_vocab_fn)
        
    # 密码子词表，用于CRF tag
    _codon_vocab_fn = os.path.join(os.path.dirname(__file__), 
                                                 '../configs/codon_vocab.json')
    if os.path.exists(_codon_vocab_fn):
        codon_vocab = load_json_file(_codon_vocab_fn)
        codon_vocab = {w: idx for idx, w in enumerate(codon_vocab)}

        
    return  inverse_trans_vocab, codon_vocab
        


"""
NOTE: CRF model tokenzier
"""

inverse_trans_vocab, codon_vocab = get_vocab()
codon_vocab_reverse = {v:k for k, v in codon_vocab.items()}
    
def _conda_mask(inputs, max_len=1024):
    """
    生成密码子mask表    
    
    params:
        input: 输入序列
        inverse_trans_vocab: aa-codon 对应表 
        codon_vocab： tags词表
        max_len: 最大序列长度
    """
    
    aa_conda_msks = np.zeros([max_len, len(codon_vocab)+2])
    for idx in range(len(inputs)):
        conda_msks = np.zeros(len(codon_vocab)+2)
        conda_msks[[codon_vocab[i] for i in inverse_trans_vocab[inputs[idx]]]] = 1 # 合法密码子
        aa_conda_msks[idx, :] = conda_msks
        
    conda_masks = torch.tensor(aa_conda_msks, dtype=torch.long)

    return conda_masks




def _tag(inputs, target, max_length=1024):
    """
    构造tag变量
    
    inputs:
        inputs: 输入氨基酸序列
        target: 目标CDS序列字符串
    return :
        tag_tensor:
        tag_mask: mask 矩阵
    """
    
    if target:
        tags = [[sentence[i:i+3] for i in range(0, len(sentence), 3)] for sentence in target]

        match_tags = [[codon_vocab[it] for it in tag] for tag in tags]
        tag_labes = [torch.tensor(tag + [codon_vocab['<pad>']]  * (
            max_length - len(tag)), dtype=torch.long) for tag in  match_tags]

        tag_tensor = torch.stack(tag_labes)
        tag_mask = (tag_tensor !=0).long()
    else:
        tag_tensor = None
        src_mask = torch.tensor([len(inp) for inp in inputs])
        tag_mask = (torch.arange(max_length, 
                            ).expand(src_mask.shape[0], -1) < src_mask.unsqueeze(1)).long()
        
    c_masks = torch.stack( [_conda_mask(list(inp), max_length) for inp in inputs])
    

    return tag_tensor, tag_mask, c_masks


def pad_tensor(input_ids, input_mask, pad, max_len):
    """
    padding value for given tensor
    """
    
    a,b = input_ids.shape
    
    input_ids = torch.cat( (input_ids, torch.ones(a, max_len-input_ids.shape[-1])*pad  ), axis=1).long()
    input_mask = torch.cat( (input_mask, torch.zeros(a, max_len-input_mask.shape[-1])), axis=1).long()

    return input_ids, input_mask
