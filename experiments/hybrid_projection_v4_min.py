#!/usr/bin/env python3
import argparse,time,math,numpy as np,torch,torch.nn as nn,torch.nn.functional as F

def seed(s):
 torch.manual_seed(s);np.random.seed(s)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
def makeW(o,i,r,d):
 r=min(r,o,i);return (torch.randn(o,r,device=d)/math.sqrt(r))@(torch.randn(r,i,device=d)/math.sqrt(i))
class Task:
 def __init__(self,s,dims,tr,n,d):
  seed(s);self.dims=dims;self.d=d;self.W=[makeW(b,a,tr,d) for a,b in zip(dims[:-1],dims[1:])]
  with torch.no_grad():
   x=torch.randn(4096,dims[0],device=d);y=self.raw(x);self.scale=y.std().clamp_min(1e-6);self.vx=torch.randn(n,dims[0],device=d);self.vy=self(self.vx)
 def raw(self,x):
  h=x
  for j,W in enumerate(self.W):
   h=h@W.t();h=torch.tanh(h) if j<len(self.W)-1 else h
  return h
 def __call__(self,x): return self.raw(x)/self.scale
 def batch(self,b):
  x=torch.randn(b,self.dims[0],device=self.d)
  with torch.no_grad(): y=self(x)
  return x,y
class Net(nn.Module):
 def __init__(self,s,dims):
  super().__init__();seed(s);self.L=nn.ModuleList([nn.Linear(a,b) for a,b in zip(dims[:-1],dims[1:])])
 def forward(self,x):
  h=x
  for j,l in enumerate(self.L):
   h=l(h);h=torch.tanh(h) if j<len(self.L)-1 else h
  return h
def params(m): return sum(p.numel() for p in m.parameters())
@torch.no_grad()
def evalv(m,t,bs):
 m.eval();v=[]
 for k in range(0,t.vx.shape[0],bs): v.append(F.mse_loss(m(t.vx[k:k+bs]),t.vy[k:k+bs]).item())
 m.train();return float(np.mean(v))
@torch.no_grad()
def grad_cache(n,rank,ema,old):
 c=[]
 for idx,l in enumerate(n.L):
  G=l.weight.grad.detach();b=l.bias.grad.detach();din=G.shape[1];r=min(rank,din);torch.manual_seed(1234+idx)
  P=torch.randn(r,din,device=G.device,dtype=G.dtype)/math.sqrt(r);C=(G@P.t())@P
  if old is not None: C=ema*old[idx][0]+(1-ema)*C;b=ema*old[idx][1]+(1-ema)*b
  c.append((C.clone(),b.clone()))
 return c
@torch.no_grad()
def micro(n,c,lr,k):
 for _ in range(k):
  for l,(G,b) in zip(n.L,c): l.weight.add_(-lr*G);l.bias.add_(-lr*b)
def step_full(n,opt,t,a):
 x,y=t.batch(a.batch);opt.zero_grad(set_to_none=True);loss=F.mse_loss(n(x),y);loss.backward();torch.nn.utils.clip_grad_norm_(n.parameters(),10.0);opt.step()
def run_dense(s,t,dims,a):
 n=Net(s+1000,dims).to(a.dev);opt=torch.optim.AdamW(n.parameters(),lr=a.lr);t0=time.time()
 for _ in range(a.steps): step_full(n,opt,t,a)
 if a.dev.type=='cuda': torch.cuda.synchronize()
 return evalv(n,t,a.eval_batch),params(n),time.time()-t0,a.steps,0
def run_anchor(s,t,dims,a):
 n=Net(s+1000,dims).to(a.dev);opt=torch.optim.AdamW(n.parameters(),lr=a.lr);c=None;t0=time.time();ms=0
 for _ in range(a.anchors):
  x,y=t.batch(a.batch);opt.zero_grad(set_to_none=True);loss=F.mse_loss(n(x),y);loss.backward();torch.nn.utils.clip_grad_norm_(n.parameters(),10.0);c=grad_cache(n,a.rank,a.ema,c);opt.step();micro(n,c,a.micro_lr,a.micro_steps);ms+=a.micro_steps
 if a.dev.type=='cuda': torch.cuda.synchronize()
 return evalv(n,t,a.eval_batch),params(n),time.time()-t0,a.anchors,ms
def run_fullplus(s,t,dims,a):
 n=Net(s+1000,dims).to(a.dev);opt=torch.optim.AdamW(n.parameters(),lr=a.lr);c=None;t0=time.time();ms=0
 for _ in range(a.steps):
  x,y=t.batch(a.batch);opt.zero_grad(set_to_none=True);loss=F.mse_loss(n(x),y);loss.backward();torch.nn.utils.clip_grad_norm_(n.parameters(),10.0);c=grad_cache(n,a.rank,a.ema,c);opt.step();micro(n,c,a.micro_lr,a.micro_steps);ms+=a.micro_steps
 if a.dev.type=='cuda': torch.cuda.synchronize()
 return evalv(n,t,a.eval_batch),params(n),time.time()-t0,a.steps,ms
def main():
 p=argparse.ArgumentParser();p.add_argument('--dims',default='64,384,384,384,64');p.add_argument('--seeds',default='0,1,2');p.add_argument('--steps',type=int,default=1800);p.add_argument('--anchors',type=int,default=450);p.add_argument('--batch',type=int,default=256);p.add_argument('--val-n',type=int,default=4096);p.add_argument('--eval-batch',type=int,default=1024);p.add_argument('--true-rank',type=int,default=48);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--rank',type=int,default=16);p.add_argument('--micro-steps',type=int,default=4);p.add_argument('--micro-lr',type=float,default=1e-4);p.add_argument('--ema',type=float,default=0.8);p.add_argument('--variants',default='dense,anchor,fullplus');p.add_argument('--threads',type=int,default=1);a=p.parse_args();torch.set_num_threads(a.threads)
 try: torch.set_num_interop_threads(1)
 except RuntimeError: pass
 a.dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');dims=[int(x) for x in a.dims.split(',')];seeds=[int(x) for x in a.seeds.split(',')];vs=[x.strip() for x in a.variants.split(',')]
 print('device',a.dev,'torch',torch.__version__,'dims',dims,'micro',a.micro_steps,'rank',a.rank,'micro_lr',a.micro_lr);rows=[]
 for s in seeds:
  t=Task(s,dims,a.true_rank,a.val_n,a.dev)
  for name,fn in [('dense',run_dense),('anchor',run_anchor),('fullplus',run_fullplus)]:
   if name not in vs: continue
   r=fn(s,t,dims,a);rows.append((name,)+r);print(f'{name:8s} seed={s} val={r[0]:.5f} params={r[1]} sec={r[2]:.2f} full={r[3]} micro={r[4]}')
 print('\nSUMMARY')
 for name in ['dense','anchor','fullplus']:
  rs=[r for r in rows if r[0]==name]
  if not rs: continue
  vals=np.array([r[1] for r in rs]);print(f'{name:8s} val={vals.mean():.5f}+/-{vals.std():.5f} params={int(np.mean([r[2] for r in rs]))} sec={np.mean([r[3] for r in rs]):.2f} full={int(np.mean([r[4] for r in rs]))} micro={int(np.mean([r[5] for r in rs]))}')
if __name__=='__main__': main()
