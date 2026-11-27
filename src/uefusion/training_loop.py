"""Selected-checkpoint training loop used for the primary domain models."""
import json
import time
import numpy as np
import torch

@torch.no_grad()
def score(model,X,y,device,domain):
    model.eval();total=0.
    for i in range(0,len(X),1024):
        xx=torch.from_numpy(X[i:i+1024]).to(device);yy=torch.from_numpy(y[i:i+1024]).to(device)
        if domain:
            p,v=model(xx,return_uncertainty=True);loss=torch.nn.functional.gaussian_nll_loss(p,yy,v,reduction='sum')
        else:
            p=model(xx);loss=torch.nn.functional.mse_loss(p,yy,reduction='sum')
        assert torch.isfinite(loss);total+=float(loss)
    return total/(2*len(X))

def fit_phase(model,X,y,Xv,yv,device,domain,seed,out,phase,epochs):
    fine=phase=='augmented'
    torch.manual_seed(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4 if fine else 5e-4,weight_decay=1e-5)
    batches=int(np.ceil(len(X)/512))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs*batches if fine else 40,eta_min=1e-6)
    best=score(model,Xv,yv,device,domain) if fine else float('inf');stale=0;logs=[]
    checkpoint=out/f'{phase}_best.pth'
    if fine:torch.save({'model_state_dict':model.state_dict(),'epoch':-1,'validation_loss':best},checkpoint)
    generator=np.random.default_rng(seed)
    for ep in range(epochs):
        t=time.monotonic();model.train();order=generator.permutation(len(X));total=0.
        for i in range(0,len(order),512):
            take=order[i:i+512];xx=torch.from_numpy(X[take]).to(device);yy=torch.from_numpy(y[take]).to(device)
            if domain:
                p,v=model(xx,return_uncertainty=True);loss=torch.nn.functional.gaussian_nll_loss(p,yy,v)
            else:loss=torch.nn.functional.mse_loss(model(xx),yy)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if fine:scheduler.step()
            total+=float(loss)*len(take)
        val=score(model,Xv,yv,device,domain)
        improved=val<best
        # Min-delta controls patience, while the minimum-loss checkpoint is retained.
        significant=val<best-.001
        if improved:best=val;torch.save({'model_state_dict':model.state_dict(),'epoch':ep,'validation_loss':best},checkpoint)
        stale=0 if significant else stale+1
        if not fine:scheduler.step()
        rec={'phase':phase,'epoch':ep,'train_loss':total/len(X),'validation_loss':val,'selected':improved,'seconds':time.monotonic()-t}
        logs.append(rec);(out/f'{phase}_history.json').write_text(json.dumps(logs,indent=2)+'\n');print(json.dumps(rec),flush=True)
        torch.save({'model_state_dict':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':ep,'phase':phase,'rng_state':generator.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(device) if device.type == 'cuda' else None},out/'last.pth')
        if not fine and stale>=10:break
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['model_state_dict'])
    return checkpoint
