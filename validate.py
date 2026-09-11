"""重新计算空间/时间收敛性和命名参数敏感性；不拟合参数。"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import argparse, csv, json
from drying_model import Config, Inputs, simulate

ROOT=Path(__file__).resolve().parent

def save_rows(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows]
    keys=list(dict.fromkeys(k for r in flat for k in r))
    with path.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as f:
        out=csv.DictWriter(f,fieldnames=keys);out.writeheader();out.writerows(flat)

def grid_study(inp, out, grids=(80,160,320)):
    rows=[]
    for case in (1,2,4):
        for n in grids:
            cfg=Config(case=case,n=n,shrink=case==4,horizon_h=.5 if case==1 else 480.,stop_at_target=case!=1)
            run=simulate(cfg,inp);met=run.metrics()
            row=dict(study='space',case=case,n=n,drying_h=met['drying_h'],
                     Tcenter_end_C=met['end_Tcenter_C'],Tsurface_end_C=met['end_Tsurface_C'],
                     Csurface_end=met['end_Csurface'],Cmax_end=met['end_Cmax'],
                     mass_residual_relative=met['max_mass_relative_residual'])
            rows.append(row);print(row,flush=True)
            del run
    # 单独考察时间误差：只改变容差和步长上限，不改变网格。
    for case in (1,2,4):
        cfg=Config(case=case,n=160,shrink=case==4,horizon_h=.5 if case==1 else 480.,stop_at_target=case!=1,
                   rtol=2e-9,atol_C=2e-11,atol_T=2e-9,max_step_early_s=7.5,max_step_late_s=90.)
        run=simulate(cfg,inp);met=run.metrics()
        row=dict(study='time_tight',case=case,n=160,drying_h=met['drying_h'],
                 Tcenter_end_C=met['end_Tcenter_C'],Tsurface_end_C=met['end_Tsurface_C'],
                 Csurface_end=met['end_Csurface'],Cmax_end=met['end_Cmax'],
                 mass_residual_relative=met['max_mass_relative_residual'])
        rows.append(row);print(row,flush=True);del run
    save_rows(Path(out)/'convergence.json',rows)
    return rows

def sensitivity_study(inp,out,n=80):
    rows=[]
    for case in (2,4):
        changes=[('baseline',1.,1.)]+[('beta',f,1.) for f in (.8,.9,1.1,1.2)]+[('Kp',1.,f) for f in (.3,3.)]
        reference=None
        for label,bf,kf in changes:
            cfg=Config(case=case,n=n,shrink=case==4,beta_factor=bf,kp_factor=kf)
            run=simulate(cfg,inp);met=run.metrics()
            if label=='baseline':reference=met['drying_h']
            row=dict(case=case,n=n,parameter=label,beta_factor=bf,Kp_factor=kf,
                     drying_h=met['drying_h'],Ceq=met['equilibrium_C'],
                     reached=met['threshold_reached'],computed_h=met['computed_end_s']/3600,
                     end_Cmax=met['end_Cmax'],
                     change_pct=None if met['drying_h'] is None else 100*(met['drying_h']/reference-1))
            rows.append(row);print(row,flush=True);del run
    save_rows(Path(out)/'sensitivity.json',rows)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',default=str(ROOT/'data'));p.add_argument('--out-dir',default=str(ROOT/'validation'))
    p.add_argument('--grid',action='store_true');p.add_argument('--sensitivity',action='store_true')
    args=p.parse_args();inp=Inputs.load(args.data_dir)
    if args.grid:grid_study(inp,args.out_dir)
    if args.sensitivity:sensitivity_study(inp,args.out_dir)
    if not(args.grid or args.sensitivity):p.error('Specify --grid and/or --sensitivity.')
