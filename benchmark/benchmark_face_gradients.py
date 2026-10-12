""" Benchmark fused normal face gradients, averaging and non-orthogonal correction.
    Checks 2D/3D cases, arbitrary bias and all normal-gradient components.
    Compares isolated kernels and full solver steps; production is unchanged.

    Run: python -m benchmark.benchmark_face_gradients
    Results: /tmp/fvm-gradient-trial-results.json
"""
import inspect
import json
import pickle
import statistics
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import triton
import triton.language as tl

from base_cfg import ARTEFACT_DIR
from time_fvm.config_fvm import ConfigEllipse
from time_fvm.fvm_equation import FVMEquation,FluidConstitution2D
from time_fvm.run_fvm import init_conds_ellipses
from time_fvm.time_solvers.integrators import get_solver
from time_fvm.fvm_stepping.facet_process import FacetCalc
from time_fvm.utils.ell_kernel import ell_spmv


@triton.autotune(configs=[
    triton.Config({'BLOCK':64},num_warps=2),
    triton.Config({'BLOCK':128},num_warps=4),
    triton.Config({'BLOCK':256},num_warps=4),
    triton.Config({'BLOCK':512},num_warps=8),
],key=['NF','C','D','K'])
@triton.jit
def gradient_kernel(U,G,Vals,Cols,Bias,Cells,XOrth,NHat,GV,GT,GN,
                    NF:tl.constexpr,C:tl.constexpr,D:tl.constexpr,K:tl.constexpr,
                    SVM:tl.constexpr,SVK:tl.constexpr,SCM:tl.constexpr,SCK:tl.constexpr,
                    SB:tl.constexpr,BLOCK:tl.constexpr):
    """ Compute normal gradients, average adjacent cell gradients and correct grad_V.
        Writes grad_V, grad_T_n and normal gradients for all components.
        NF = n_facets, C = n_comp, D = dim, K = ELL entries per row.

        U: Us, shape = [n_cells, n_comp]
        G: limited cell_grads, shape = [n_cells, dim, n_comp]
        Vals, Cols: A_face_grad, shape = [n_facets*n_comp, K]
        Bias: b_face_grad, shape = [n_facets*n_comp]
        Cells: face_cell_indices, shape = [n_facets, 2]
        XOrth, NHat: X_orthog, normals_hat, shape = [n_facets, dim, 1]
        GV: grad_V, shape = [n_facets, dim, dim]
        GT: grad_T_n, shape = [n_facets]
        GN: normal face gradients, shape = [n_facets, n_comp]
    """
    i = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    mask = i < NF*C
    facet,comp = i//C,i%C
    # Direct face gradient: grad_faces_n = A_face_grad @ Us.flatten() + b_face_grad
    # Preserve all components, including density
    dn = tl.zeros((BLOCK,),tl.float32)
    for kk in range(K):
        value = tl.load(Vals+i*SVM+kk*SVK,mask=mask,other=0.)
        col = tl.load(Cols+i*SCM+kk*SCK,mask=mask,other=0)
        u = tl.load(U+col,mask=mask,other=0.)
        dn += value*u
    dn += tl.load(Bias+i*SB,mask=mask,other=0.)
    tl.store(GN+i,dn,mask=mask)
    # Average limited cell gradients: grad_F_lstsq = 0.5 * (grad_L + grad_R)
    # Boundary facets use the adjacent cell on both sides
    grad_mask = mask & (comp != D)
    left = tl.load(Cells+2*facet,mask=grad_mask,other=0)
    right = tl.load(Cells+2*facet+1,mask=grad_mask,other=0)
    averages,normals = (),()
    orth,dot_n = tl.zeros((BLOCK,),tl.float32),tl.zeros((BLOCK,),tl.float32)
    for d in tl.static_range(D):
        gl = tl.load(G+left*D*C+d*C+comp,mask=grad_mask,other=0.)
        gr = tl.load(G+right*D*C+d*C+comp,mask=grad_mask,other=0.)
        average = .5*(gl+gr)
        x = tl.load(XOrth+facet*D+d,mask=grad_mask,other=0.)
        n = tl.load(NHat+facet*D+d,mask=grad_mask,other=0.)
        averages += (average,)
        normals += (n,)
        orth += average*x    # grad_F_lstsq . X_orthog
        dot_n += average*n   # grad_F_lstsq . normals_hat
    # Correct normal derivative: dFdn_correct = grad_faces_n - grad_F_lstsq . X_orthog
    correction = dn-orth
    # Replace the normal part, retaining the tangential part:
    # grad_F = grad_F_lstsq + (dFdn_correct - grad_F_lstsq . normals_hat) * normals_hat
    for d in tl.static_range(D):
        value = averages[d]+(correction-dot_n)*normals[d]
        tl.store(GV+facet*D*D+d*D+comp,value,mask=mask & (comp<D))
    # Temperature normal derivative for conduction: grad_T_n = dFdn_correct[T]
    tl.store(GT+facet,correction,mask=mask & (comp==D+1))


def fused(fc,U,grads):
    """ Allocate outputs and launch the experimental face-gradient kernel. """
    mesh = fc.mesh
    op = mesh.A_face_grad
    gv = torch.empty(fc.n_facets,fc.dim,fc.dim,device=U.device)
    gt = torch.empty(fc.n_facets,device=U.device)
    gn = torch.empty(fc.n_facets,fc.n_comp,device=U.device)
    grid = lambda meta:(triton.cdiv(fc.n_facets*fc.n_comp,meta['BLOCK']),)
    gradient_kernel[grid](U,grads,op.vals,op.cols,mesh.b_face_grad,
        mesh.face_cell_indices,mesh.X_orthog,mesh.normals_hat,gv,gt,gn,
        fc.n_facets,fc.n_comp,fc.dim,op.vals.shape[1],
        op.vals.stride(0),op.vals.stride(1),op.cols.stride(0),op.cols.stride(1),mesh.b_face_grad.stride(0))
    return gv,gt,gn


def original(fc,U,grads):
    """ Reference calculation from FacetCalc.compute_facet. """
    mesh,D = fc.mesh,fc.dim
    selected = grads[:,:,fc.grad_comps].reshape(fc.n_cells,D*(D+1))
    avg = .5*(selected[mesh.face_cell_indices[:,0]]+selected[mesh.face_cell_indices[:,1]])
    avg = avg.reshape(fc.n_facets,D,D+1)
    gn = fc._face_grads(U)
    corr = gn[:,fc.grad_comps]-(avg*mesh.X_orthog).sum(dim=1)
    dot = (avg*mesh.normals_hat).sum(dim=1,keepdim=True)
    corrected = avg+(corr.unsqueeze(1)-dot)*mesh.normals_hat
    return corrected[:,:,:D],corr[:,-1],gn


def check(actual,expected,label,atol=2e-6):
    details = []
    for a,b in zip(actual,expected):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        torch.testing.assert_close(a,b,rtol=2e-5,atol=atol)
        details.append({'max_abs':(a-b).abs().max().item(),
            'relative_l2':(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-30)).item()})
    print('CHECK '+label+' '+json.dumps(details),flush=True)
    return details


def synthetic():
    for D in (2,3):
        N,NF,C = 37,43,D+2
        U = torch.randn(N,C,device='cuda')
        grads = torch.randn(N,D,C,device='cuda')
        cells = torch.randint(N,(NF,2),device='cuda',dtype=torch.int32)
        cells[-7:,1] = cells[-7:,0]
        nhat = torch.randn(NF,D,1,device='cuda')
        nhat /= torch.linalg.vector_norm(nhat,dim=1,keepdim=True)
        for K in (0,1,2,3,5):
            vals = torch.randn(NF*C,K,device='cuda')
            cols = torch.randint(N*C,(NF*C,K),device='cuda',dtype=torch.int32)
            bias = torch.randn(NF*C*2,device='cuda')[::2]
            op = SimpleNamespace(vals=vals,cols=cols)
            mesh = SimpleNamespace(A_face_grad=op,b_face_grad=bias,face_cell_indices=cells,
                                   X_orthog=torch.randn(NF,D,1,device='cuda'),normals_hat=nhat)
            fc = SimpleNamespace(mesh=mesh,dim=D,n_cells=N,n_facets=NF,n_comp=C,
                                 grad_comps=list(range(D))+[D+1])
            fc._face_grads = lambda U,vals=vals,cols=cols,bias=bias:ell_spmv(vals,cols,U.flatten(),bias).view(NF,C)
            check(fused(fc,U,grads),original(fc,U,grads),'synthetic D='+str(D)+' K='+str(K))


def install(fc):
    """ Replace the face-gradient block on the after benchmark instance only. """
    source = textwrap.dedent(inspect.getsource(FacetCalc.compute_facet))
    start = source.index('    # Compute face and cell gradients')
    source = source[:start]+'    self.grad_V, self.grad_T_n, normal_grads = fused(self, Us, cell_grads)\n'
    ns = dict(FacetCalc.compute_facet.__globals__,fused=fused,__name__=__name__)
    exec(compile(source,'<fused-face-gradients>','exec'),ns)
    fc.compute_facet = types.MethodType(ns['compute_facet'],fc)


def build(saved):
    np.random.seed(321)
    torch.manual_seed(321)
    cfg = ConfigEllipse()
    cfg.compile = False
    physics = FluidConstitution2D(cfg,dim=2)
    tags,state = init_conds_ellipses(saved['mesh'],saved['edge_tag'],saved['bound_edgs'],physics,cfg)
    eq = FVMEquation(cfg,physics,saved['mesh'],tags,Us_init=state)
    solver = get_solver(eq,cfg)
    solver._solve_step = torch.compile(solver._solve_step,fullgraph=True)
    return solver


def graph(fn):
    for _ in range(3):fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20):fn()
    return g


def graph_time(g):
    a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(30):g.replay()
    b.record();b.synchronize()
    return a.elapsed_time(b)*1000/600


def measure(solver,t):
    torch.cuda.synchronize()
    a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(100):solver._solve_step(t)
    b.record();b.synchronize()
    return a.elapsed_time(b)/100


@torch.inference_mode()
def main():
    torch.manual_seed(321)
    synthetic()
    with open(ARTEFACT_DIR/'fvm_mesh.pkl','rb') as stream:saved=pickle.load(stream)
    solvers = {name:build(saved) for name in ('before','after')}
    fc = solvers['before'].eq.E_props
    U = solvers['before'].cells.get_values()[0]
    U += torch.randn_like(U)*U.new_tensor([.1,.1,.01,1.])
    fc.compute_facet(U)
    grads = fc.cell_grads
    functions = {'before':torch.compile(lambda:original(fc,U,grads),fullgraph=True),
                 'after':torch.compile(lambda:fused(fc,U,grads),fullgraph=True)}
    actual,baseline = functions['after'](),functions['before']()
    # Absolute tolerance for near-zero velocity gradients after cancellation.
    isolated_check = check(actual,baseline,'actual mesh compiled',atol=2e-5)
    torch.testing.assert_close(actual[2],baseline[2],rtol=0,atol=0)
    # Compare both float32 paths against the same double-precision equations.
    mesh,D = fc.mesh,fc.dim
    gd = grads.double()[:,:,fc.grad_comps]
    avg = .5*(gd[mesh.face_cell_indices[:,0]]+gd[mesh.face_cell_indices[:,1]])
    op = mesh.A_face_grad
    normal64 = (op.vals.double()*U.double().flatten()[op.cols]).sum(dim=1)+mesh.b_face_grad.double()
    normal64 = normal64.view(fc.n_facets,fc.n_comp)
    corr64 = normal64[:,fc.grad_comps]-(avg*mesh.X_orthog.double()).sum(dim=1)
    dot64 = (avg*mesh.normals_hat.double()).sum(dim=1,keepdim=True)
    gv64 = (avg+(corr64.unsqueeze(1)-dot64)*mesh.normals_hat.double())[:,:,:D]
    precision = {}
    for name,values in [('before',baseline),('after',actual)]:
        difference = values[0].double()-gv64
        precision[name] = {'max_abs_velocity_gradient_error':difference.abs().max().item(),
                          'relative_l2':(torch.linalg.vector_norm(difference)/torch.linalg.vector_norm(gv64)).item()}
    print('FLOAT64 REFERENCE '+json.dumps(precision),flush=True)
    del gd,avg,normal64,corr64,dot64,gv64
    graphs = {name:graph(fn) for name,fn in functions.items()}
    timings = {name:[] for name in graphs}
    for repeat in range(9):
        for name in (list(graphs) if repeat%2==0 else list(graphs)[::-1]):timings[name].append(graph_time(graphs[name]))
    isolated = {name:statistics.median(ds) for name,ds in timings.items()}
    print('ISOLATED '+json.dumps(isolated),flush=True)
    del graphs,functions
    # Restore previous gradients so both full solvers start identically.
    fc.cell_grads = None
    install(solvers['after'].eq.E_props)
    t = torch.tensor(0.,device='cuda')
    for name,s in solvers.items():
        print('Compiling/warming '+name,flush=True)
        for _ in range(20):s._solve_step(t)
        torch.cuda.synchronize()
    snapshots = {name:(s.cells.state.clone(),s.eq.E_props.cell_grads.clone(),s.dt.clone()) for name,s in solvers.items()}
    matched = {}
    for name,s in solvers.items():
        state,g,dt = snapshots['before']
        s.cells.state,s.eq.E_props.cell_grads = state.clone(),g.clone()
        s.dt.copy_(dt);s._solve_step(t)
        matched[name] = s.cells.state.clone()
    single = check((matched['after'],),(matched['before'],),'matched full step')
    for name,s in solvers.items():
        state,g,dt = snapshots[name]
        s.cells.state,s.eq.E_props.cell_grads = state,g
        s.dt.copy_(dt)
    full_samples = {name:[] for name in solvers}
    for repeat in range(5):
        for name in (list(solvers) if repeat%2==0 else list(solvers)[::-1]):
            timing = measure(solvers[name],t)
            full_samples[name].append(timing)
            print(json.dumps({'repeat':repeat,'variant':name,'gpu_ms':timing}),flush=True)
    medians = {name:statistics.median(ds) for name,ds in full_samples.items()}
    assert all(torch.isfinite(s.cells.state).all() for s in solvers.values())
    for s in solvers.values():
        state,g,dt = snapshots['before']
        s.cells.state,s.eq.E_props.cell_grads = state.clone(),g.clone()
        s.dt.copy_(dt)
    for _ in range(500):
        dt = solvers['before'].dt.clone()
        solvers['before']._solve_step(t)
        solvers['after'].dt.copy_(dt);solvers['after']._solve_step(t)
    a,b = solvers['before'].cells.state,solvers['after'].cells.state
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    result = {'gpu':torch.cuda.get_device_name(),'cells':fc.n_cells,'facets':fc.n_facets,
        'isolated_us':isolated,'isolated_samples':timings,'full_step_ms':medians,'full_samples':full_samples,
        'isolated_check':isolated_check,'float64_reference_errors':precision,'single_step_check':single,
        'same_dt_relative_l2_per_component':(torch.linalg.vector_norm(a-b,dim=0)/torch.linalg.vector_norm(a,dim=0)).tolist(),
        'reduction_percent':100*(1-medians['after']/medians['before'])}
    Path('/tmp/fvm-gradient-trial-results.json').write_text(json.dumps(result,indent=2))
    print('RESULT '+json.dumps(result),flush=True)


if __name__ == '__main__':main()
