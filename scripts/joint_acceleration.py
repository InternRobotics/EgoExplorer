"""World-joint acceleration on irregular timestamps; first 22 body joints."""
import torch


def joint_acceleration_loss(joints, times, frame_valid):
    if len(times) < 3:
        return joints.sum() * 0
    dt = times[1:] - times[:-1]
    valid_step = (dt > 0) & (dt <= .1)
    keep = valid_step[1:] & valid_step[:-1]
    keep = keep & frame_valid[:-2] & frame_valid[1:-1] & frame_valid[2:]
    safe_dt = dt.clamp_min(1e-6)
    velocity = (joints[1:, :22] - joints[:-1, :22]) / safe_dt[:, None, None]
    acceleration = 2 * (velocity[1:] - velocity[:-1]) / (safe_dt[1:] + safe_dt[:-1])[:, None, None]
    squared = acceleration.square().mean(dim=(1, 2))
    return torch.where(keep, squared, torch.zeros_like(squared)).sum() / keep.sum().clamp_min(1)


def preflight(device='cpu'):
    t=torch.tensor([0., .01, .028, .045, .06],dtype=torch.float64,device=device)
    valid=torch.ones(5,dtype=torch.bool,device=device)
    linear=(2*t[:,None,None]).expand(-1,22,3).clone().requires_grad_()
    assert joint_acceleration_loss(linear,t,valid).item()<1e-20
    quadratic=(3*t[:,None,None]**2).expand(-1,22,3).clone().requires_grad_()
    loss=joint_acceleration_loss(quadratic,t,valid)
    assert abs(loss.item()-36)<1e-9
    loss.backward()
    assert torch.isfinite(quadratic.grad).all() and quadratic.grad.abs().sum()>0
    invalid=torch.zeros_like(valid)
    assert joint_acceleration_loss(quadratic,t,invalid).item()==0
    gap=torch.tensor([0.,.02,1.],dtype=torch.float64,device=device)
    assert joint_acceleration_loss(quadratic[:3],gap,valid[:3]).item()==0
    return dict(device=device,constant_velocity_zero=True,constant_acceleration_exact=True,
                finite_gradients=True,invalid_and_gap_excluded=True)

if __name__=='__main__':
    import json,sys
    print(json.dumps(preflight(sys.argv[1] if len(sys.argv)>1 else 'cpu')))
