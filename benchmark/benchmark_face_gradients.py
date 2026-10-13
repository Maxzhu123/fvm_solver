""" Benchmark fused normal face gradients, averaging and non-orthogonal correction.
    Checks 2D/3D cases, arbitrary bias and all normal-gradient components.
    Compares the integrated kernel against the PyTorch reference, including full steps.

    Run: python -m benchmark.benchmark_face_gradients
    Results: /tmp/fvm-gradient-trial-results.json
"""
import json
import pickle
import statistics
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from base_cfg import ARTEFACT_DIR
from time_fvm.config_fvm import ConfigEllipse
from time_fvm.fvm_equation import FVMEquation,FluidConstitution2D
from time_fvm.run_fvm import init_conds_ellipses
from time_fvm.time_solvers.integrators import get_solver
from time_fvm.fvm_stepping.facet_process import FacetCalc
from time_fvm.utils.ell_kernel import ell_spmv
from time_fvm.utils.face_gradient_kernel import compute_face_gradients


def fused(fc,U,grads):
    """ Launch the integrated face-gradient kernel. """
    return compute_face_gradients(fc.mesh,U,grads)


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

        # Exercise the production CPU fallback with nonzero boundary bias.
        U_cpu,grads_cpu = U.cpu(),grads.cpu()
        vals_cpu,cols_cpu,bias_cpu = vals.cpu(),cols.cpu(),bias.cpu()
        fc.mesh = SimpleNamespace(face_cell_indices=cells.cpu(),X_orthog=mesh.X_orthog.cpu(),
                                  normals_hat=nhat.cpu())
        fc._face_grads = lambda U:(vals_cpu*U.flatten()[cols_cpu]).sum(1).view(NF,C)+bias_cpu.view(NF,C)
        check(FacetCalc._compute_face_gradients(fc,U_cpu,grads_cpu),original(fc,U_cpu,grads_cpu),
              'CPU fallback D='+str(D))


def install_reference(fc):
    """ Use the PyTorch reference on the before benchmark instance only. """
    fc._compute_face_gradients = types.MethodType(original,fc)


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
    install_reference(solvers['before'].eq.E_props)
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
