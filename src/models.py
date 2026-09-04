
"""
AI vs Human Text Detection - complete practical implementation
Rules respected for the from-scratch section:
- ML algorithms implemented manually with NumPy.
- No scikit-learn/TensorFlow/Keras/PyTorch used in the NumPy models.
PyTorch section implements DNN, LSTM and a Transformer encoder.
"""

import re, json, random, pickle
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -------------------------
# Text processing / TF-IDF
# -------------------------
def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?", str(text).lower())

class TfidfVectorizerNumpy:
    def __init__(self, max_features=3000, min_df=1, max_df=1.0):
        self.max_features=max_features; self.min_df=min_df; self.max_df=max_df

    def fit(self, texts):
        docs=[tokenize(x) for x in texts]
        n=len(docs); df={}
        for toks in docs:
            for t in set(toks): df[t]=df.get(t,0)+1
        vocab=[t for t,c in df.items() if c>=self.min_df and c/n<=self.max_df]
        vocab.sort(key=lambda t:(-df[t],t))
        vocab=vocab[:self.max_features]
        self.vocab={t:i for i,t in enumerate(vocab)}
        self.idf=np.array([np.log((n+1)/(df[t]+1))+1 for t in vocab], dtype=np.float32)
        return self

    def transform(self, texts):
        X=np.zeros((len(texts),len(self.vocab)), dtype=np.float32)
        for i,text in enumerate(texts):
            toks=tokenize(text)
            if not toks: continue
            counts={}
            for t in toks:
                if t in self.vocab: counts[t]=counts.get(t,0)+1
            total=sum(counts.values()) or 1
            for t,c in counts.items(): X[i,self.vocab[t]]=(c/total)*self.idf[self.vocab[t]]
            norm=np.linalg.norm(X[i])
            if norm>0: X[i]/=norm
        return X

def stratified_split(df, train=0.70, val=0.15, seed=42):
    rng=np.random.default_rng(seed)
    parts=[]
    for label,g in df.groupby("label"):
        idx=np.arange(len(g)); rng.shuffle(idx)
        n=len(g); ntr=int(round(n*train)); nv=int(round(n*val))
        parts += [(g.iloc[idx[:ntr]],"train"),(g.iloc[idx[ntr:ntr+nv]],"val"),(g.iloc[idx[ntr+nv:]],"test")]
    tr=pd.concat([x for x,s in parts if s=="train"]).sample(frac=1,random_state=seed).reset_index(drop=True)
    va=pd.concat([x for x,s in parts if s=="val"]).sample(frac=1,random_state=seed).reset_index(drop=True)
    te=pd.concat([x for x,s in parts if s=="test"]).sample(frac=1,random_state=seed).reset_index(drop=True)
    return tr,va,te

def metrics(y,p):
    y=np.asarray(y).astype(int); p=np.asarray(p).astype(int)
    tp=((y==1)&(p==1)).sum(); tn=((y==0)&(p==0)).sum()
    fp=((y==0)&(p==1)).sum(); fn=((y==1)&(p==0)).sum()
    acc=(tp+tn)/max(len(y),1); prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1)
    f1=2*prec*rec/max(prec+rec,1e-12)
    return {"accuracy":float(acc),"precision":float(prec),"recall":float(rec),"f1":float(f1),
            "tp":int(tp),"tn":int(tn),"fp":int(fp),"fn":int(fn)}

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-50,50)))

class LogisticRegressionNumpy:
    def __init__(self, lr=0.05, epochs=500, l2=1e-4):
        self.lr=lr; self.epochs=epochs; self.l2=l2
    def fit(self,X,y,Xv=None,yv=None):
        n,d=X.shape; self.W=np.zeros(d); self.b=0.; self.history=[]
        for e in range(self.epochs):
            z=X@self.W+self.b; p=sigmoid(z)
            gradW=(X.T@(p-y))/n + self.l2*self.W
            gradb=np.mean(p-y)
            self.W-=self.lr*gradW; self.b-=self.lr*gradb
            loss=-np.mean(y*np.log(p+1e-9)+(1-y)*np.log(1-p+1e-9))+self.l2*np.sum(self.W**2)/2
            self.history.append(loss)
        return self
    def predict_proba(self,X): return sigmoid(X@self.W+self.b)
    def predict(self,X): return (self.predict_proba(X)>=.5).astype(int)

class DenseNumpy:
    def __init__(self, in_dim, hidden=(128,32), lr=1e-3, epochs=250, batch=16, dropout=.25, l2=1e-4, patience=25):
        self.layers=[in_dim,*hidden,1]; self.lr=lr; self.epochs=epochs; self.batch=batch
        self.dropout=dropout; self.l2=l2; self.patience=patience
        self.W=[]; self.b=[]
        for a,b in zip(self.layers[:-1],self.layers[1:]):
            self.W.append(np.random.randn(a,b).astype(np.float32)*np.sqrt(2/a)); self.b.append(np.zeros(b,dtype=np.float32))
    def fit(self,X,y,Xv,yv):
        y=y.reshape(-1,1).astype(np.float32); yv=yv.reshape(-1,1).astype(np.float32)
        best=None; best_loss=1e9; wait=0; self.history={"loss":[],"val_loss":[]}
        for ep in range(self.epochs):
            idx=np.random.permutation(len(X))
            for s in range(0,len(X),self.batch):
                ids=idx[s:s+self.batch]; A=X[ids].astype(np.float32); Y=y[ids]
                acts=[A]; masks=[]
                for k in range(len(self.W)-1):
                    Z=acts[-1]@self.W[k]+self.b[k]; A=np.maximum(0,Z)
                    if self.dropout:
                        m=(np.random.rand(*A.shape)>=self.dropout).astype(np.float32)/(1-self.dropout)
                        A*=m; masks.append(m)
                    else: masks.append(None)
                    acts.append(A)
                P=sigmoid(acts[-1]@self.W[-1]+self.b[-1]); dz=P-Y
                dW=[None]*len(self.W); db=[None]*len(self.b)
                dW[-1]=acts[-1].T@dz/len(Y)+self.l2*self.W[-1]; db[-1]=dz.mean(0)
                da=dz@self.W[-1].T
                for k in range(len(self.W)-2,-1,-1):
                    if masks[k] is not None: da*=masks[k]
                    dz=da*(acts[k+1]>0)
                    dW[k]=acts[k].T@dz/len(Y)+self.l2*self.W[k]; db[k]=dz.mean(0)
                    da=dz@self.W[k].T
                for k in range(len(self.W)): self.W[k]-=self.lr*dW[k]; self.b[k]-=self.lr*db[k]
            trp=self.predict_proba(X); vap=self.predict_proba(Xv)
            tl=-np.mean(y*np.log(trp+1e-9)+(1-y)*np.log(1-trp+1e-9)); vl=-np.mean(yv*np.log(vap+1e-9)+(1-yv)*np.log(1-vap+1e-9))
            self.history["loss"].append(float(tl)); self.history["val_loss"].append(float(vl))
            if vl<best_loss-1e-5:
                best_loss=vl; best=([w.copy() for w in self.W],[b.copy() for b in self.b]); wait=0
            else: wait+=1
            if wait>=self.patience: break
        self.W,self.b=best
        return self
    def predict_proba(self,X):
        A=X.astype(np.float32)
        for k in range(len(self.W)-1): A=np.maximum(0,A@self.W[k]+self.b[k])
        return sigmoid(A@self.W[-1]+self.b[-1]).ravel()
    def predict(self,X): return (self.predict_proba(X)>=.5).astype(int)

# -------------------------
# PyTorch models
# -------------------------
def build_torch_models(vocab_size, pad_idx=0, max_len=128):
    import torch
    import torch.nn as nn

    class TorchDNN(nn.Module):
        def __init__(self, d):
            super().__init__(); self.net=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,1))
        def forward(self,x): return self.net(x).squeeze(-1)

    class LSTMClassifier(nn.Module):
        def __init__(self,vocab,emb=64,h=64):
            super().__init__(); self.emb=nn.Embedding(vocab,emb,padding_idx=pad_idx); self.lstm=nn.LSTM(emb,h,batch_first=True,bidirectional=True); self.drop=nn.Dropout(.3); self.fc=nn.Linear(h*2,1)
        def forward(self,x):
            e=self.emb(x); out,(h,c)=self.lstm(e); z=torch.cat([h[-2],h[-1]],1); return self.fc(self.drop(z)).squeeze(-1)

    class TinyTransformer(nn.Module):
        def __init__(self,vocab,emb=64,nhead=4,layers=2):
            super().__init__(); self.emb=nn.Embedding(vocab,emb,padding_idx=pad_idx); self.pos=nn.Embedding(max_len,emb)
            enc=nn.TransformerEncoderLayer(d_model=emb,nhead=nhead,dim_feedforward=128,dropout=.2,batch_first=True)
            self.enc=nn.TransformerEncoder(enc,num_layers=layers); self.fc=nn.Linear(emb,1)
        def forward(self,x):
            B,L=x.shape; pos=torch.arange(L,device=x.device).unsqueeze(0).expand(B,L); h=self.emb(x)+self.pos(pos)
            mask=x.eq(pad_idx); h=self.enc(h,src_key_padding_mask=mask); valid=(~mask).unsqueeze(-1); pooled=(h*valid).sum(1)/valid.sum(1).clamp(min=1)
            return self.fc(pooled).squeeze(-1)
    return TorchDNN,LSTMClassifier,TinyTransformer

def make_sequence_vocab(texts, min_freq=1, max_vocab=8000, max_len=128):
    from collections import Counter
    toks=[tokenize(t) for t in texts]; c=Counter(x for z in toks for x in z)
    words=[w for w,n in c.most_common() if n>=min_freq][:max_vocab-2]
    vocab={"<PAD>":0,"<UNK>":1}; vocab.update({w:i+2 for i,w in enumerate(words)})
    def enc(t):
        ids=[vocab.get(w,1) for w in tokenize(t)][:max_len]
        return ids+[0]*(max_len-len(ids))
    return vocab,enc

if __name__=="__main__":
    print("Import this module from the notebooks. See notebooks/ for the complete workflow.")
