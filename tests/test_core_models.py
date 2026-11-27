import numpy as np
import pytest
import torch
from uefusion.models import make_model
from uefusion.training_loop import score
from uefusion.boosting import fit_axis


@pytest.mark.parametrize('name,n',[('ran',35),('core',15),('central44',44),('central42',42)])
def test_architecture_outputs(name,n):
    torch.set_num_threads(1);torch.manual_seed(42);m=make_model(name).eval()
    with torch.no_grad():
        if name in ('ran','core'):
            p,v=m(torch.zeros(3,10,n),return_uncertainty=True)
            assert p.shape==v.shape==(3,2) and torch.isfinite(p).all()
            assert torch.all(v>=np.exp(-4)) and torch.all(v<=np.exp(10))
        else:assert m(torch.zeros(3,10,n)).shape==(3,2)


def test_gaussian_nll_normalization():
    p=torch.tensor([[1.,2.],[2.,3.]]);y=torch.zeros_like(p);v=torch.tensor([[2.,4.],[3.,5.]])
    actual=torch.nn.functional.gaussian_nll_loss(p,y,v)
    expected=(torch.log(v)+(p-y)**2/v).sum()/(4*len(p))
    torch.testing.assert_close(actual,expected)


def test_gb_selects_validation_best_not_final(tmp_path):
    rng=np.random.default_rng(8);x=rng.normal(size=(60,4));y=x[:,0]*2+x[:,1];xv=rng.normal(size=(25,4));yv=xv[:,0]*2+xv[:,1]
    model,info=fit_axis(x,y,xv,yv,42,tmp_path,0,rule=dict(chunk=5,patience=2,cap=15))
    assert model.n_iter_==info['best_iter']
    assert info['best_iter']==5*(int(np.argmin(info['val_mse_per_chunk']))+1)
    assert np.mean((model.predict(xv)-yv)**2)==pytest.approx(info['best_val_mse'])
