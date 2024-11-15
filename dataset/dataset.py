# coding=utf-8

"""
Implementation of a SMILES dataset.
"""

import torch
import torch.utils.data as tud
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from dataset.tokenizer import Tokenizer, _tag


# class Dataset(tud.Dataset):
#     """Custom PyTorch Dataset that takes a file containing \n separated SMILES"""

#     def __init__(self, smiles_list, vocabulary, tokenizer, device, max_length=0):
#         self._vocabulary = vocabulary
#         self._tokenizer = tokenizer
#         self._smiles_list = smiles_list # list(smiles_list)
#         self.max_length = max_length
#         self.device=device

#     def __getitem__(self, i):
#         # smi = self._smiles_list[i]
#         smi = next(iter(self._smiles_list))['smiles']
#         print(f'smiles: {smi}')
#         tokens = self._tokenizer.tokenize(smi)
#         encoded = self._vocabulary.encode(tokens)
#         return torch.tensor(encoded, dtype=torch.long, device=self.device)  # pylint: disable=E1102

#     def __len__(self):
#         return len(self._smiles_list)

#     @staticmethod
#     def collate_fn(encoded_seqs):
#         """Converts a list of encoded sequences into a padded tensor"""
#         max_length = max([seq.size(0) for seq in encoded_seqs])
#         collated_arr = torch.zeros(len(encoded_seqs), max_length, dtype=torch.long, device=encoded_seqs[0].device)  # padded with zeroes
#         for i, seq in enumerate(encoded_seqs):
#             collated_arr[i, :seq.size(0)] = seq
#         return collated_arr


def calculate_nlls_from_model(model, smiles, batch_size=128):
    """
    Calculates NLL for a set of SMILES strings.
    :param model: Model object.
    :param smiles: List or iterator with all SMILES strings.
    :return : It returns an iterator with every batch.
    """
    dataset = Dataset(smiles, model.vocabulary, model.tokenizer)
    _dataloader = tud.DataLoader(dataset, batch_size=batch_size, collate_fn=Dataset.collate_fn)

    def _iterator(dataloader):
        for batch in dataloader:
            nlls = model.likelihood(batch.long())
            yield nlls.data.cpu().numpy()

    return _iterator(_dataloader), len(_dataloader)





class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.to(device)

    def __len__(self):
        return self.data.size(0) // self.seq_len

class TextSamplerIterDataset(IterableDataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __iter__(self):

        for i in range(self.data.size(0) // self.seq_len):
            
            rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
            full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()

            yield full_seq.to(device)


class CrfDataset(Dataset):
    """
    CRF SFT dataset process
    """
    def __init__(self, data, max_seq_length=None, device='cpu'):
        super().__init__()
        self.data = data
        self.max_seq_length = max_seq_length
        self.tokenizer = Tokenizer()
        self.device = device

    def __getitem__(self, index):

        line = self.data[index]

        xb, yb = line['aa'][:self.max_seq_length], line['cds'][:self.max_seq_length*3]
        _input_ids = self.tokenizer.tokenize_to_ids(xb, seq_type='aa')
        input_ids = _input_ids[:self.max_seq_length] + [0.]*(self.max_seq_length-len(_input_ids)) 
        tag_tensor, tag_mask, c_masks = _tag([xb], [yb], max_length=self.max_seq_length)
        return torch.tensor(input_ids, device=self.device).long(), tag_tensor.squeeze(0), tag_mask.squeeze(0), c_masks.squeeze(0)


    def __len__(self):
        return len(self.data)


class PromptDataset():
    """
    prompts process
    """
    def __init__(self, data, max_seq_length=None, device='cpu'):
        super().__init__()
        self.data = data
        self.max_seq_length = max_seq_length
        self.tokenizer = Tokenizer()
        self.device = device

    def tensor(self):

        xb = self.data[:self.max_seq_length]
        _input_ids = self.tokenizer.tokenize_to_ids(xb, seq_type='aa')
        input_ids = _input_ids[:self.max_seq_length] + [0.]*(self.max_seq_length-len(_input_ids)) 
        _, tag_mask, c_masks = _tag([xb], None, max_length=self.max_seq_length)
        return torch.tensor(input_ids, device=self.device).long().unsqueeze(0), tag_mask, c_masks


    def __len__(self):
        return len(self.data)



class CrfIterDataset(IterableDataset):
    def __init__(self, data, max_seq_length):
        super().__init__()
        self.data = data
        self.max_seq_length = max_seq_length
    def __iter__(self):

        for line in self.data:
            # print("lines:", line['smiles'])
            print("lines:", line)
            # tokens = tokenizer.tokenize(line['smiles'])
            tokens = tokenizer.tokenize(line)
            encoded = vocabulary.encode(tokens)
            pad_encoded= list(encoded)+ [0.]*(self.max_seq_length-len(encoded)) if len(encoded) < self.max_seq_length else encoded[:self.max_seq_length]
            # examples['smiles'] = torch.tensor(pad_encoded, device=device).long()
            yield torch.tensor(pad_encoded, device=device).long()


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            max_seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        max_seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        map_func = None,
        content_field="tokens",
        accelerator = None,
        data_size = None
    ):
        self.tokenizer = tokenizer
        # self.concat_token_id = tokenizer.bos_token_id
        self.dataset = dataset
        self.max_seq_length = max_seq_length
        self.epoch = 0
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = max_seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.map_func = map_func
        self.accelerator = accelerator
        self.data_size = data_size

    def __iter__(self):
        if self.map_func:
            self.dataset = map(self.map_func, self.dataset)

        # sample to buffer size
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    sample = next(iterator)
                    buffer.append(sample[self.content_field])
                    buffer_len += len(buffer[-1])
                    # self.accelerator.print(f"Dataset buffer_len: {buffer_len}")
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        self.epoch += 1
                        # if self.accelerator:
                        #     self.accelerator.print(f"Dataset epoch: {self.epoch}")
                        # else:
                        #     print(f"Dataset epoch: {self.epoch}")
      
                    else:
                        more_examples = False
                        break
            # tokenizer and prepare for inputs
            tokenized_inputs = self.tokenizer(buffer)
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input)

            for i in range(0, len(all_token_ids), self.max_seq_length):
                input_ids = all_token_ids[i : i + self.max_seq_length]
                if len(input_ids) == self.max_seq_length:
                    if max(input_ids) > 256 or min(input_ids) < 0:
                        print("wrong input_ids index: ", input_ids)
                        continue
                    self.current_size += 1
                    yield torch.tensor(input_ids).long()

    def __len__(self):
        return self.data_size # 3610000 # 965696 # 1000  # len(self.dataset) 965696

    def shuffle(self, buffer_size=1000):
        return ShufflerIterDataPipe(self, buffer_size=buffer_size)