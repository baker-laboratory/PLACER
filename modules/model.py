from typing import List,Dict,Any
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import dgl
import sys, os

sys.path.append(os.path.dirname(__file__))
from SE3_network import SE3TransformerWrapper
import geometry
import losses
from attention import PairStr2Pair,FeedForwardLayer


# ============================================================
def get_grads(X         : torch.Tensor,
              bonds     : torch.Tensor,
              bondlen   : torch.Tensor,
              chirals   : torch.Tensor,
              planars   : torch.Tensor,
              gclip     : float = 100.0) -> torch.Tensor:
    '''get gradients of bonded restraints w.r.t. atom positions
    
    Args:
        X:        [L,3], atom coordinates
        bonds:    [M,3], pairs of atoms forming bonds + reference bond values
        chirals:  [O,5], quadruples of atoms forming chiral centers
        planars:  [P,5], quadruples of atoms forming planar groups
        gclip:    gradients are clipped by norm using this cutoff

    Returns:
        clipped gradients of the above three restraints w.r.t. X
    '''
    
    L = X.shape[0]
    device = X.device

    # enable autograd, no matter what

    with torch.enable_grad():

        Xg = X.detach()
        Xg.requires_grad = True

        g = torch.zeros((L,3,3), device=device)

        # (0) bonds
        if bonds.shape[0]>0:
            l = losses.bondLoss(X=Xg, ij=bonds, b0=bondlen, mean=False)
            g[:,0] = torch.autograd.grad(l,Xg)[0].data

        # (1) chirals
        if chirals.shape[0]>0:
            o,i,j,k = Xg[chirals].permute(1,0,2)
            l = ((geometry.triple_prod(o-i,o-j,o-k,norm=True)-0.70710678)**2).sum()
            g[:,1] = torch.autograd.grad(l,Xg)[0].data

        # (2) planars
        if planars.shape[0]>0:
            o,i,j,k = Xg[planars].permute(1,0,2)
            l = (geometry.triple_prod(o-i,o-j,o-k,norm=True)**2).sum()
            g[:,2] = torch.autograd.grad(l,Xg)[0].data


    # scale & clip
    g = torch.nan_to_num(g, nan=0.0, posinf=gclip, neginf=-gclip)
    gnorm = torch.linalg.norm(g, dim=-1)
    mask = gnorm>gclip
    g[mask] /= gnorm[mask][...,None]
    g[mask] *= gclip

    return g.detach()


# ============================================================
def make_topk_graph(xyz  : torch.Tensor,
                    D    : torch.Tensor,
                    sep  : torch.Tensor,
                    mask : torch.Tensor,
                    topk : int = 32) -> torch.Tensor:
    '''
    Input:
        - xyz: atom cooordinates (L, 3, 3)
        - pair: pair features (L, L, E)
        - sep: bond separation matrix
    Output:
        - G: dgl graph
    '''

    L = xyz.shape[0]
    device = xyz.device

    # 1) select up to topk//2 neighbors based on bond separation
    # (we are adding +1 to topk to account for the self-edge which is deleted later)
    _,idx = torch.topk(sep.masked_fill(sep==0,999), min(topk//2+1,L), dim=1, largest=False)
    sep_mask = torch.zeros_like(sep,dtype=bool).scatter_(1,idx,True)
    sep_mask = sep_mask&(sep>0)

    # 2) get the remaining topk neighbors based on proximity in 3D space
    D_with_sep = D.masked_fill(sep_mask,0.0)
    _,idx = torch.topk(D_with_sep, min(topk+1, L), largest=False)
    dist_mask = torch.zeros_like(D,dtype=bool).scatter_(1,idx,True)
    
    # 3) get edges, create a DGL graph
    cond = dist_mask&mask
    i,j = torch.where(cond.fill_diagonal_(False)) # self-edges are deleted here

    # dgl not supported on MPS. Move tebsors to CPU if device is MPS
    if xyz.is_mps:
        i = i.to("cpu")
        j = j.to("cpu")
        xyz_cpu = xyz.to("cpu")
        G = dgl.graph((i, j), num_nodes=L).to("cpu")
        G.edata['rel_pos'] = (xyz_cpu[j] - xyz_cpu[i]).detach()
    else:
        G = dgl.graph((i, j), num_nodes=L).to(device)
        G.edata['rel_pos'] = (xyz[j] - xyz[i]).detach() # no gradients through basis functions

    return G


# ============================================================
def rbf(D: torch.Tensor,
        d_rbf : float = 12.0,
        n_rbf : int = 16) -> torch.Tensor:
    '''decompose distances into radial basis functions'''
    
    sigma = d_rbf / n_rbf
    mu = torch.linspace(sigma, d_rbf, n_rbf, device=D.device)
    rbf = torch.exp(-((D[...,None]-mu) / sigma)**2)

    return rbf


# ============================================================
class InitEmbedder1D(nn.Module):
    '''simple embedding layer to process 1D inputs'''
    
    def __init__(self, 
                 d_in : int,
                 d_hidden : int):

        super(InitEmbedder1D, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden)
        )

    def forward(self, f1d):
        return self.net(f1d)

    
# ============================================================
class InitEmbedder2D(nn.Module):
    '''simple embedding layer to process 1D and 2D inputs'''
    
    def __init__(self, 
                 d_f1d : int,
                 d_f2d : int,
                 d_hidden : int):

        super(InitEmbedder2D, self).__init__()
        
        self.left = nn.Linear(d_f1d, d_hidden)
        self.right = nn.Linear(d_f1d, d_hidden)
        self.linear1 = nn.Linear(d_f2d, d_hidden)
        self.linear2 = nn.Linear(2*d_hidden, d_hidden)
        self.norm = nn.LayerNorm(d_hidden)

    def forward(self, f1d, f2d):
        
        L = f1d.shape[0]
        left = self.left(f1d)
        right = self.right(f1d)
        left_right = left[:,None,:] + right[None,:,:]
        pair = self.linear1(f2d)
        pair = torch.cat([pair,left_right], dim=-1)
        pair = self.norm(self.linear2(F.relu_(pair)))

        return pair

    
# ============================================================
class PLACER_network(nn.Module):
    ''''''
    
    def __init__(self, 
                 SE3_PARAMS : Dict[str,int],
                 PAIR_PARAMS : Dict[str,Any],
                 dims1d : List[int],
                 dims2d : List[int],
                 d_rbf : float = 20.0,
                 n_rbf : int = 32,
                 topk : int = 32,
                 gclip : float = 100.0,
                 l1_scale : float = 100.0,
                 gradchk : bool = True):
        ''' '''
        super(PLACER_network, self).__init__()
        
        assert SE3_PARAMS['l0_in_features']==SE3_PARAMS['l0_out_features']
        
        d_node = SE3_PARAMS['l0_in_features']
        d_edge = SE3_PARAMS['num_edge_features']
        d_pair = PAIR_PARAMS['d_pair']
        p_drop = PAIR_PARAMS['p_drop']
        
        SE3_PARAMS['l1_in_features'] = 3
        SE3_PARAMS['l1_out_features'] = 1

        self.gclip = gclip
        self.l1_scale = l1_scale
        self.gradchk = gradchk
        self.topk = topk
        
        d_f1d = sum(dims1d)
        d_f2d = sum(dims2d)

        self.rbf = lambda X : rbf(X, d_rbf=d_rbf, n_rbf=n_rbf)
        
        self.init_single = InitEmbedder1D(d_f1d, d_node)
        self.init_pair = InitEmbedder2D(d_f1d,d_f2d, d_pair)
        self.pair_to_edge = FeedForwardLayer(d_pair, d_edge, r_ff=2, p_drop=0.0, normalize=False)
        
        self.embed_pair = PairStr2Pair(**PAIR_PARAMS)
        self.update_pair = PairStr2Pair(**PAIR_PARAMS)
        self.update_single = FeedForwardLayer(d_node, d_node, r_ff=2, p_drop=p_drop, normalize=True)

        self.se3 = SE3TransformerWrapper(**SE3_PARAMS)

        self.norm_single = nn.LayerNorm(d_node)
        self.norm_pair = nn.LayerNorm(d_pair)

        # predicted per-atom lddts
        self.predict_plddt = nn.Sequential(
            nn.LayerNorm(d_node),
            nn.Linear(d_node,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, 51)
        )

        # predicted per-atom deviations
        self.predict_dev = nn.Sequential(
            nn.LayerNorm(d_node),
            nn.Linear(d_node,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.ReLU()
        )

        # predicted distance error head
        self.predict_pde = nn.Sequential(
            nn.LayerNorm(d_pair),
            nn.Linear(d_pair,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, 102)
        )


    def forward(self, X,f1d,f2d,separation,bonds,bondlen,chirals,planars,recycles,mask=None,save_recycling_data=False):

        L = X.shape[0]
        if mask is None:
            mask = torch.full((L,L),True, dtype=bool, device=X.device)

        # unwrap SE3 outputs to allow for gradient checkpointing
        se3 = lambda *inputs : tuple(self.se3(*inputs).values())
        
        # initial embeddings
        single = self.init_single(f1d)
        pair   = self.init_pair(f1d,f2d)
        
        # update pair embeddings with the 3D structure info
        D = torch.cdist(X,X)
        D = D.masked_fill(~mask,999.9).fill_diagonal_(999.9)
        rbf = self.rbf(D).detach()
        pair = self.embed_pair(pair[None], rbf[None], mask[None])[0]

        # return values
        Xs = [X]
        PDEs = []
        plDDTs = []
        pDEVs = []

        # previous values for skip-connections
        single_prev,pair_prev = None,None

        #
        # run recycles
        #
        for i_rec,flag in enumerate(recycles):

            if single_prev is not None:
                single = self.norm_single(single_prev)
                pair   = self.norm_pair(pair_prev)
            else:
                single_prev = single
                pair_prev = pair

            # get gradients of bonded restraints
            l1 = get_grads(X=Xs[-1],
                           bonds=bonds,
                           bondlen=bondlen,
                           chirals=chirals,
                           planars=planars,
                           gclip=self.gclip).detach()

            # build graph
            G = make_topk_graph(xyz=Xs[-1],
                                D=D,
                                sep=separation,
                                mask=mask,
                                topk=self.topk)

            # run se3 transformer backpropping only
            # through the flagged iterations
            nodes = single[...,None]
            edges = pair[G.edges()]
            if flag!=0 and self.training==True:
                if self.gradchk==True:
                    edges = checkpoint.checkpoint(self.pair_to_edge, edges)[...,None]
                    state,dx = checkpoint.checkpoint(se3, G, nodes, l1, edges)
                else:
                    edges = self.pair_to_edge(edges)[...,None]
                    state,dx = se3(G, nodes, l1, edges)
            else:
                with torch.no_grad():
                    edges = self.pair_to_edge(edges)[...,None]
                    state,dx = se3(G, nodes, l1, edges)

            # for MPS, dgl operations are moved to the CPU due to the lack of MPS support
            # thus, state and dx are currently on the CPU. Move them back to MPS
            if X.is_mps:
                state = state.to('mps')
                dx = dx.to('mps')

            # update coordinates
            X_new = Xs[-1].detach() + dx[:,0]/self.l1_scale
            Xs.append(X_new)
            
            # update state features
            state = state.reshape(single.shape)
            single = self.update_single(state+single)

            # update pair features; run auxiliary heads
            D = torch.cdist(X_new,X_new)
            D = D.masked_fill(~mask,999.9).fill_diagonal_(999.9)
            rbf = self.rbf(D).detach()
            if flag!=0 and self.training==True:
                if self.gradchk==True:
                    rbf.requires_grad = True
                    pair = checkpoint.checkpoint(self.update_pair, pair[None], rbf[None], mask[None])[0]
                    pde = checkpoint.checkpoint(self.predict_pde, pair+pair.permute(1,0,2))
                    plddt = checkpoint.checkpoint(self.predict_plddt,single)
                    dev = checkpoint.checkpoint(self.predict_dev,single)
                else:
                    pair = self.update_pair(pair[None], rbf[None], mask[None])[0]
                    pde = self.predict_pde(pair+pair.permute(1,0,2))
                    plddt = self.predict_plddt(single)
                    dev = self.predict_dev(single)
            else:  # inference run
                with torch.no_grad():
                    pair = self.update_pair(pair[None], rbf[None], mask[None])[0]

                    ## To save GPU memory, these metrics are only calculated/stored at the last recycle step
                    ## Unless user requests full recycling output
                    if save_recycling_data == True or i_rec == len(recycles)-1:
                        pde = self.predict_pde(pair+pair.permute(1,0,2))
                        plddt = self.predict_plddt(single)
                        dev = self.predict_dev(single)

            # skip-connections
            single_prev = single_prev + single
            pair_prev   = pair_prev + pair

        # save predictions from auxiliary heads
        PDEs.append(pde)
        plDDTs.append(plddt)
        pDEVs.append(dev[...,0])

        return Xs,PDEs,plDDTs,pDEVs
