import torch as th
from icecream import ic


if __name__ == "__main__":
    a = th.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=th.float32)
    a = a.reshape(5,2)
    
    d = th.tensor([0,0,1,0,1], dtype=th.float32).reshape(5,-1)
    ic(a)
    
    b = th.hstack((a,d))
    ic(b)
    cond = th.all(b,dim=-1, keepdim=True)
    ic(cond)
    
    zeros = th.zeros(3, dtype=th.float32).reshape(1,-1)
    zeros[:,-1] = 1
    ic(zeros)
    
    ans=th.where(cond, zeros, b)
    ic(ans)