
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import make_scorer, recall_score, confusion_matrix
import sklearn.metrics as metrics
from torch.nn import functional as F
from torch import einsum, nn
import torch
import collections
import math


from nltk.translate.bleu_score import sentence_bleu,  corpus_bleu
import warnings
from Bio import BiopythonDeprecationWarning
warnings.simplefilter('ignore', BiopythonDeprecationWarning)

from Bio import SeqIO
import Bio.SeqUtils.CodonUsage
from Bio import pairwise2 as pw2
from Bio.Seq import Seq


def r2_loss(output, target):

    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2



class RMSELoss(nn.Module):
    """
    wraper of MSE
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss



def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
def NPV(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]/(confusion_matrix(y_true, y_pred)[0, 0] + confusion_matrix(y_true, y_pred)[1, 0])

# specificity = make_scorer(recall_score, pos_label=0)

# skl_binary_scoring={'tp': make_scorer(tp), 'tn': make_scorer(tn),
#          'fp': make_scorer(fp), 'fn': make_scorer(fn),
#          'accuracy': metrics.accuracy_score, 'recall': metrics.recall_score, 'precision': metrics.precision_score,
#          'PPV': metrics.precision_score,'NPV': make_scorer(NPV),
#           'sensitivity': metrics.recall_score,'specificity': specificity,
#          'f1': metrics.f1_score,'auc': metrics.roc_auc_score}

binary_scoring={'tp': tp, 'tn': tn,
         'fp': fp, 'fn': fn,
         'accuracy': metrics.accuracy_score, 'recall': metrics.recall_score, 'precision': metrics.precision_score,
         'PPV': metrics.precision_score,'NPV': NPV,
          'sensitivity': metrics.recall_score,
         'f1': metrics.f1_score,'auc': metrics.roc_auc_score}


def binary_metrics(y_trues, y_preds, y_probs):
    """ 
    calculate score for binary class problem
    """
    metrics_res = {}
    for k, v in binary_scoring.items():
        if k == 'auc':
            metrics_res[k] = v(y_trues, y_probs)
        else:
            metrics_res[k] = v(y_trues, y_preds)

    return metrics_res


def regression_metrics(y_preds, y_trues):
    """
    calculate regression score
    """
    
    MSE = F.mse_loss(y_preds, y_trues).item()

    MAE = F.l1_loss(y_preds, y_trues).item()

    RMSE = RMSELoss()(y_preds, y_trues).item()

    R2 = r2_loss(y_preds, y_trues).item()
    
    return {
        'MSE': MSE,
        'RMSE': RMSE,
        'MAE': MAE,
        'R2': R2
    
    }



def bleu(pred_seq, label_seq, k):  #@save
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

def bleu_score(pred_seq, label_seq, target_vocab_reverse):
    """
    计算bleu分值
    """
    # 只计算有效位置字符
    # reference = [[target_vocab_reverse[k.item()] for k in all_token_ids[i][:int(valid_lens[i])].flatten()] for i in range(all_token_ids.shape[0])]

    # candidate = [[target_vocab_reverse[k.item()] for k in Y_hat[i][:int(valid_lens[i])].flatten()] for i in range(Y_hat.shape[0])]

    references = [[target_vocab_reverse[k.item()] for k in label_seq[i].flatten()] for i in range(label_seq.shape[0])]

    candidates = [[target_vocab_reverse[k.item()] for k in pred_seq[i].flatten()] for i in range(pred_seq.shape[0])]

    return corpus_bleu(references, candidates)



def calc_similarity(seq1, seq2, penalty=False):
    """
    计算序列全局比对相似性
    params:
        seq: sgrings of protein or nuclitides
        penalty: whether using gap/dismatch penalty
    """
    
    if penalty:
        global_align = pw2.align.globalms(Seq(seq1), Seq(seq2), 2, -1, -.5, -.1)
    else:
        global_align = pw2.align.globalxx(Seq(seq1), Seq(seq2))
    
    if not global_align:
        percent_match = 0
    else:
        
        matched = global_align[0]
        aligned_len = len(matched.seqA)
        identical_positions = sum(a == b for a,b in zip(matched.seqA, matched.seqB))
        percent_match = (identical_positions / aligned_len)
    
    return percent_match



def evaluate(true_tags, pred_tags):
    """
    评价准则
    """
    from itertools import chain
    from sklearn import metrics
    
    
    true_label = [x[x !=0].tolist() for x in torch.cat([*true_tags])]
    pred_label = [i for item in pred_tags for i in item]
    y_true = list(chain.from_iterable(true_label))
    y_pred = list(chain.from_iterable(pred_label))

    acc = metrics.accuracy_score(y_true, y_pred)

    precision = metrics.precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = metrics.recall_score(y_true, y_pred, average='macro', zero_division=1)
    
    return acc, precision, recall


def ump_evaluate(y_true, y_pred):
    """
    评价准则
    """
    from itertools import chain
    from sklearn import metrics

    acc = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = metrics.recall_score(y_true, y_pred, average='macro', zero_division=1)
    
    return acc, precision, recall