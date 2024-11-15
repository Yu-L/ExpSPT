
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models import SPT
from models.utils import multinomial_logits,gumbel_logits, gumbel_sample, gumbel_samples, top_p as top_P, top_k as top_K


"""
实现CRF层

"""



def log_sum_exp(x):
    """calculate log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """
    max_score = x.max(-1)[0]
    return max_score + (x - max_score.unsqueeze(-1)).exp().sum(-1).log()


def top_k_top_p_filtering(logits, top_k=None, top_p=None, prob=False,
                            temperature=0.1, k=1, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """

    _logits = logits.clone()
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k:
        logits = top_K(logits, k = top_k, dim=-1)

    if top_p:
        logits = top_P(logits, thres = top_p)

    # Sample from the filtered distribution
    # sample = gumbel_samples(logits, temperature = 1, dim = -1)
    sample = gumbel_logits(logits, temperature = 1, dim = -1, prob=prob)
    # sample = multinomial_logits(logits, _logits, temperature = 0.8, dim = -1)

    return sample



IMPOSSIBLE = -1e4
    
    
class CrfLM(nn.Module):
    """
    
    氨基酸CRF语言模型任务
    AA--CDS 映射关系的学习
    
    NOTE: 
    对氨基酸序列进行人工分词以确保CDS区域CRF的分词正确性
    
    ref: https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
    https://github.com/jidasheng/bi-lstm-crf/tree/master
    
    """
    
    def __init__(self, tagset_size, hidden_dim):
        super(CrfLM, self).__init__()
     
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        
        self.num_tags = tagset_size + 2
        self.start_idx = self.num_tags - 2
        self.stop_idx = self.num_tags - 1

        self.fc = nn.Linear(hidden_dim, self.num_tags)
        
    
        # transition factor, Tij mean transition from j to i
        self.transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags), requires_grad=True)
        self.transitions.data[self.start_idx, :] = IMPOSSIBLE
        self.transitions.data[:, self.stop_idx] = IMPOSSIBLE
   
    def forward(self, features, ys, masks, inference=False, c_masks=None):
        """negative log likelihood loss
        B: batch size, L: sequence length, D: dimension
        params:
            :param features: [B, L, D]
            :param ys: tags, [B, L]
            :param masks: masks for padding, [B, L]
            :inference : inference mode
        :return: loss
        """
        features = self.fc(features)
        
        B, L, C = features.shape
        masks_ = masks[:, :L].float()
        
        gold_score, best_score, best_paths = None, None, None
        
        # # 添加一个conda_mask 用于mask掉非法密码子
        if c_masks is not None:
            c_masks = c_masks[:, :L, :].float()
            features = features * c_masks.to(features.device) + \
                (1-c_masks.to(features.device)) *torch.full((B, L, C), IMPOSSIBLE, device=features.device)
                
        forward_score = self.__forward_algorithm(features, masks_, c_masks, strict=False) # 模型预测值

        if ys is not None:
            gold_score = self.__score_sentence(features, ys[:, :L].long(), masks_, c_masks, strict=False) # 真实label
   
        if inference:
            best_score, best_paths = self.__viterbi_decode(features, masks, c_masks, strict=False)
  
            
        return forward_score, gold_score, best_score, best_paths

    def generate(self, features, masks, c_masks=None, temperature=0.1, top_k=2, sample=False, strict=False):
        """negative log likelihood loss
        B: batch size, L: sequence length, D: dimension
        params:
            :param features: [B, L, D]
            :param ys: tags, [B, L]
            :param masks: masks for padding, [B, L]
            :inference : inference mode
        :return: loss
        """
        features = self.fc(features)
        
        B, L, C = features.shape
        masks_ = masks[:, :L].float()
        
        best_score, best_paths = None, None
        
        # # 添加一个conda_mask 用于mask掉非法密码子
        if c_masks is not None:
            c_masks = c_masks[:, :L, :].float()
            features = features * c_masks.to(features.device) + \
                (1-c_masks.to(features.device)) *torch.full((B, L, C), IMPOSSIBLE, device=features.device)
                
        best_score, best_paths = self.__viterbi_decode(features, masks, c_masks, strict=strict, temperature=temperature, top_k=top_k, sample=sample)
            
        return best_score, best_paths
   
    def __score_sentence(self, features, tags, masks, c_masks=None, strict=False):
        """Gives the score of a provided tag sequence
        :param features: [B, L, C]
        :param tags: [B, L]
        :param masks: [B, L]
        :return: [B] score in the log space
        """
        B, L, C = features.shape
        
        # emission score
        emit_scores = features.gather(dim=2, index=tags.unsqueeze(-1)).squeeze(-1)
        # transition score
        start_tag = torch.full((B, 1), self.start_idx, dtype=torch.long, device=tags.device)
        tags = torch.cat([start_tag, tags], dim=1)  # [B, L+1]
        
        # del start_tag
        trans_scores = self.transitions[tags[:, 1:], tags[:, :-1]]

        # NOTE: 当且仅当batch（B）为1时，直接在self.transitions 上 mask，batch！=1 时，还没有想好批量mask的矩阵运算方法
        # # 添加一个conda_mask 用于mask掉非法密码子
        if c_masks is not None and strict:
            # 添加对transitions的个性化mask
            conda_transi = self.transitions.repeat(B, L, 1, 1)
            # 对所有batch构造
            conda_transi = c_masks.to(features.device).unsqueeze(3).transpose(2,3) * conda_transi \
                                + (1-c_masks.to(features.device).unsqueeze(3).transpose(2,3)) * \
                                    torch.full((B, L, 1, 1), IMPOSSIBLE, device=features.device)
            # 对每一个batch，每条序列的每一个pos计算个性化mask后的转移分数
            for b in range(B):
                for t in range(L):
                    trans_scores[b, t] = conda_transi[b, t, ...][ tags[b, t+1], tags[b, t]]
        
        # last transition score to STOP tag
        last_tag = tags.gather(dim=1, index=masks.sum(1).long().unsqueeze(1)).squeeze(1)  # [B]
        last_score = self.transitions[self.stop_idx, last_tag]
        # del last_tag

        score = ((trans_scores + emit_scores) * masks).sum(1) + last_score

        return score


    def __viterbi_decode(self, features, masks, c_masks=None, strict=False, temperature=0.1, top_k=2, sample=False):
        """decode to tags using viterbi algorithm
        :param features: [B, L, C], batch of unary scores
        :param masks: [B, L] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, L]
        """
        B, L, C = features.shape
                
        bps = torch.zeros(B, L, C, dtype=torch.long, device=features.device)  # back pointers

        # Initialize the viterbi variables in log space
        max_score = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
        max_score[:, self.start_idx] = 0
        
        # # 添加一个conda_mask 用于mask掉非法密码子
        if c_masks is not None and strict:
            max_score = max_score - max_score*c_masks.to(features.device)[:, 0,:]
            # 添加对transitions的个性化mask
            conda_transi = self.transitions.repeat(B, L, 1, 1)
            # 对所有batch构造
            conda_transi = c_masks.to(features.device).unsqueeze(3).transpose(2,3) * conda_transi \
                                + (1-c_masks.to(features.device).unsqueeze(3).transpose(2,3)) *torch.full((B, L, 1, 1), IMPOSSIBLE, device=features.device)
        
        for t in range(L):
            mask_t = masks[:, t].unsqueeze(1).float()  # [B, 1]
            emit_score_t = features[:, t]  # [B, C]
            
            # mask for codon
            if c_masks is not None and strict: 
                acc_score_t = (max_score*c_masks.to(features.device)[:, t,:]).unsqueeze(1) + conda_transi[:, t, ...] # [B, 1, C] + [B, C, C]
            else:
                acc_score_t = max_score.unsqueeze(1) + self.transitions  # [B, 1, C] + [C, C] = [B, C, C]
                
            if sample:
                acc_score_t, bps[:, t, :], _ = top_k_top_p_filtering(acc_score_t, temperature=temperature, top_k=top_k)
                # print(f"after acc_score_t: {acc_score_t.shape}, acc_score_t_tag.shape: {bps[:, t, :].shape}, acc_score_t_tag: {bps[:, t, :]}")
      
            else:
                acc_score_t, bps[:, t, :] = acc_score_t.max(dim=-1)
                # print(f"after acc_score_t: {acc_score_t.shape}, acc_score_t_tag.shape: {bps[:, t, :].shape}, acc_score_t_tag: {bps[:, t, :]}")

            acc_score_t += emit_score_t
            max_score = acc_score_t * mask_t + max_score * (1 - mask_t)  # max_score or acc_score_t
        
        # Transition to STOP_TAG
        if c_masks is not None and strict:
            max_score += self.transitions[self.stop_idx]* c_masks[:, -1, :].to(features.device) \
                            + (1-c_masks[:, -1, :].to(features.device)) *torch.full((B, C), IMPOSSIBLE, device=features.device) # 此处为最后一个pos的conda转stop idx, 也需要mask
        else: 
            max_score += self.transitions[self.stop_idx]
            

        if sample:
            best_score, best_tag, probs = top_k_top_p_filtering(max_score, temperature=temperature, top_k=top_k, prob=True)
  
        else:
            best_score, best_tag = max_score.max(dim=-1)
            probs = F.softmax(max_score, dim=-1).take(best_tag).squeeze(0)

        # Follow the back pointers to decode the best path.
        best_paths = []
        bps = bps.cpu().numpy()
        for b in range(B):

            best_tag_b = best_tag[b].item()
            seq_len = int(masks[b, :].sum().item())
        
            best_path = [best_tag_b]
            for bps_t in reversed(bps[b, :seq_len]):
                best_tag_b = bps_t[best_tag_b]
                best_path.append(best_tag_b)
            
            # drop the last tag and reverse the left
            best_paths.append(best_path[-2::-1])
                #             # 丢掉最开始，并反序
                #             best_path.pop(0)
                #             best_paths.append(best_path[::-1])
        return probs, best_paths


    def __forward_algorithm(self, features, masks, c_masks=None, strict=False):
        """calculate the partition function with forward algorithm.
        TRICK: log_sum_exp([x1, x2, x3, x4, ...]) = log_sum_exp([log_sum_exp([x1, x2]), log_sum_exp([x3, x4]), ...])
        :param features: features. [B, L, C]
        :param masks: [B, L] masks
        :return:    [B], score in the log space
        """ 
        B, L, C = features.shape
        
        scores = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
        scores[:, self.start_idx] = 0
        trans = self.transitions.unsqueeze(0)  # [1, C, C]
        
        if c_masks is not None and strict:
            # 添加对transitions的个性化mask
            conda_transi = self.transitions.repeat(B, L, 1, 1)
            # 对所有batch构造
            conda_transi = c_masks.to(features.device).unsqueeze(3).transpose(2,3) * conda_transi \
                                + (1-c_masks.to(features.device).unsqueeze(3).transpose(2,3)) *torch.full((B, L, 1, 1), IMPOSSIBLE, device=features.device)
        
        # Iterate through the sentence
        for t in range(L):
            emit_score_t = features[:, t].unsqueeze(2)  # [B, C, 1]
            
            if c_masks is not None and strict:
                score_t = (scores*c_masks.to(features.device)[:, t,:]).unsqueeze(1) + conda_transi[:, t, ...] + emit_score_t  # 对score进行mask，[B, 1, C] + [B, C, C] + [B, C, 1] => [B, C, C]
            else:
                score_t = scores.unsqueeze(1) + trans + emit_score_t  # [B, 1, C] + [1, C, C] + [B, C, 1] => [B, C, C]
            
            score_t = log_sum_exp(score_t)  # [B, C]
            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            scores = score_t * mask_t + scores * (1 - mask_t)
        # del score_t
        # del mask_t
        # print("scores: ", scores[0,0])
        # print("scores.shape: ", scores.shape)
        if c_masks is not None and strict:
            scores = log_sum_exp(scores + self.transitions[self.stop_idx]* c_masks[:, -1,:].to(features.device) \
                                + (1-c_masks[:, -1, :].to(features.device)) *torch.full((B, C), IMPOSSIBLE, device=features.device))
        else:
            scores = log_sum_exp(scores + self.transitions[self.stop_idx])
        # print("after scores: ", scores)
        return scores
    




class SPT_CRF(nn.Module):
    """
    
    spt + CRF
    
    NLP pretrained mBART with a CRF layer after
    
    to solve and improve CDS tag task presion and correction
    """
    
    def __init__(self, tagset_size, hidden_dim, model_weights=None, freeze=False):
        super(SPT_CRF, self).__init__()

        self.model_weights = model_weights 
        n_heads = n_layers = 32
        dim = hidden_dim = 2048
    
        self.spt = SPT(
            num_tokens= 101,
            dim=dim,
            depth=n_layers,
            dim_head = dim//n_heads ,
            heads = n_heads,
            flash_attn=True
        )
        if self.model_weights:
            self.spt.load_state_dict(
                        torch.load(self.model_weights, weights_only=True))
        
        # freeze pretrain model's parameters
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        self.CRF = CrfLM(tagset_size, hidden_dim)
        
    def forward(self, 
                input_ids, 
                tag_tensor=None,
                tag_mask=None,
                c_masks=None):
            
        with torch.cuda.amp.autocast():
            outputs = self.spt(input_ids, return_logits_with_embedding=True)  # forward pass

        decoder_hidden_states = outputs[-1]

        forward_score, gold_score, best_score, best_paths = self.CRF(decoder_hidden_states, 
                                                                    tag_tensor, 
                                                                    tag_mask, inference=True, c_masks=c_masks)


        return forward_score, gold_score, best_score, best_paths
    
    def generate(self, input_ids, tag_mask, c_masks=None, temperature=0.1, top_k=2, sample=False, strict=False):
        
        with torch.cuda.amp.autocast():
            outputs = self.spt(input_ids, return_logits_with_embedding=True)  # forward pass

        # for CDS generate
        decoder_hidden_states = outputs[-1]
        best_score, best_paths = self.CRF.generate(decoder_hidden_states, tag_mask, c_masks=c_masks,
                                                     temperature=temperature, top_k=top_k, sample=sample, strict=strict)

        # for UTR


        return  best_score, best_paths


    def load_weights(self, model_weights=None):
 
        own_state = self.state_dict()
        state_dict = torch.load(self.model_weights, weights_only=True)
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)