
import os
import random
import gzip
import warnings
from Bio import BiopythonDeprecationWarning, BiopythonExperimentalWarning
warnings.simplefilter('ignore', BiopythonDeprecationWarning)
from Bio.Seq import Seq

"""
数据预处理脚本

mRNA.fa 格式：

>ENSMUST00000070533.4|ENSMUSG00000051951.5|Xkr4|3634|UTR5:1-150|CDS:151-2094|UTR3:2095-3634|
GGACAGGTGTCAGATAAAGGAG

注释行依次标注出 转录本id|基因ID|基因名|序列长度|UTR5坐标|CDS坐标|UTR3坐标|

过滤掉所有不具有标准mRNA结构的序列
"""



def _load_mRNA(file_name, max_len=None, compress=False, shuffle=False, filter=True):
    """
    读取并处理mRNA的fasta
    
    fasta 格式:
        > 基因名|序列长度|UTR5坐标|CDS坐标|UTR3坐标|
        ATGGGGG
        
    返回：[UTR5, CDS, UTR3, AA]
    """

    if compress:
        with gzip.open(file_name, mode='rt', encoding='utf-8') as f:
            lines = f.readlines()
    else:
        with open(file_name, 'r') as f:
            lines = f.readlines()

    mRNAs = []
    for line in lines:
        if line.startswith('>'):
                
            utr5_regs, cds_regs, utr3_regs, label = None, None, None, None
            ids = line.strip()
            for reg in  ids.split('|'):
                if 'UTR5:' in reg:
                    utr5_regs = reg.split(':')[-1].split('-')
                if 'CDS:' in reg:
                    cds_regs = reg.split(':')[-1].split('-')
                if 'UTR3:' in reg:
                    utr3_regs = reg.split(':')[-1].split('-')
                if 'LABEL:' in reg:
                    label = reg.split(':')[-1]
                    
            regs = [utr5_regs, cds_regs, utr3_regs]       
            
        else: 
            utr5, cds, utr3 = [line.strip().upper()[int(reg[0])-1:int(reg[1])] if reg is not None else '' for reg in regs ]
            
            # 仅考虑标准起始终止密码子
            if filter:
                if len(cds) % 3 != 0:
                    continue
                # if cds[:3] != 'ATG':
                #     continue
                # if cds[-3:] not in ['TAG', 'TGA', 'TAA']:
                #     continue
                
            if max_len and  (len(utr5) + len(cds) + len(utr3))  > max_len:
                continue

            if len(line) < 10:
                continue
            
            mRNAs.append([ids, line.strip().upper(), utr5, cds, utr3, str(Seq(cds).translate()), label])
            
            
    if shuffle:
        random.shuffle(mRNAs)
        
    return mRNAs