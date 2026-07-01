import argparse, time
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

# Minimal adaptive-rank projected network.
# True backprop trains both task weights and the low-rank projection structure.

def make_task(seed, width=384, true_rank=48):
    torch.manual_seed(seed); dims=[64,width,width,width,64]; layers=[]
    for a,b in zip(dims[:-1],dims[1:]):
        W=torch.randn(b,a); U,S,Vh=torch.linalg.svd(W,full_matrices=False); S[true_rank:]=0
        layers.append((U*S)@Vh)
    with torch.no_grad():
        x=torch.randn(4096,64); y=teacher(x,layers); layers[-1]/=y.std().clamp_min(1e-6)
    return layers,dims

def teacher(x,layers):
    h=x
    for i,W in enumerate(layers):
        h=h@W.t(); h=torch.tanh(h) if i<len(layers)-1 else h
    return h

def gen(layers,batch):
    x=torch.randn(batch,64)
    with torch.no_grad(): y=teacher(x,layers)
    return x,y

class AdaptiveRankNet(nn.Module):
    def __init__(self, seed, dims, r0=8, rmax=96):
        super().__init__(); torch.manual_seed(seed); self.dims=dims; self.r=[r0]*(len(dims)-1); self.rmax=rmax
        self.U=nn.ParameterList([nn.Parameter(torch.randn(dims[i+1],r0)/(r0**0.5)) for i in range(len(dims)-1)])
        self.V=nn.ParameterList([nn.Parameter(torch.randn(r0,dims[i])/(dims[i]**0.5)) for i in range(len(dims)-1)])
        self.b=nn.ParameterList([nn.Parameter(torch.zeros(dims[i+1])) for i in range(len(dims)-1)])
    def forward(self,x):
        for i in range(len(self.r)):
            x=(x@self.V[i].t())@self.U[i].t()+self.b[i]
            x=torch.tanh(x) if i<len(self.r)-1 else x
        return x
    def grow(self,i,add=8):
        add=min(add,self.rmax-self.r[i])
        if add<=0: return False
        self.U[i]=nn.Parameter(torch.cat([self.U[i].data,torch.randn(self.dims[i+1],add)*0.01],1))
        self.V[i]=nn.Parameter(torch.cat([self.V[i].data,torch.randn(add,self.dims[i])*0.01],0))
        self.r[i]+=add; return True

def params(m): return sum(p.numel() for p in m.parameters())
def evaluate(net,layers,batch=128,n=8):
    vals=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=gen(layers,batch); vals.append(F.mse_loss(net(x),y).item())
    return float(np.mean(vals))

def run(seed,steps,batch,budget):
    layers,dims=make_task(seed); net=AdaptiveRankNet(seed+50,dims); opt=torch.optim.Adam(net.parameters(),lr=1e-3); t0=time.time()
    for t in range(steps):
        x,y=gen(layers,batch); opt.zero_grad(); loss=F.mse_loss(net(x),y); loss.backward(); grew=False
        if t>0 and t%200==0 and sum(net.r)<budget:
            scores=[(net.U[i].grad.norm()+net.V[i].grad.norm()).item()/(net.r[i]**0.5) for i in range(len(net.r))]
            grew=net.grow(int(np.argmax(scores)), min(8,budget-sum(net.r)))
        if grew: opt=torch.optim.Adam(net.parameters(),lr=1e-3)
        else: opt.step()
    return evaluate(net,layers,batch,12),params(net),net.r,time.time()-t0

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=3000); p.add_argument('--batch',type=int,default=128); p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]); p.add_argument('--budget',type=int,default=160)
    a=p.parse_args(); losses=[]
    for s in a.seeds:
        loss,par,r,t=run(s,a.steps,a.batch,a.budget); losses.append(loss)
        print(f'seed {s}: loss={loss:.4f} params={par} ranks={r} time={t:.1f}s')
    print(f'mean loss={np.mean(losses):.4f} +/- {np.std(losses):.4f}')
