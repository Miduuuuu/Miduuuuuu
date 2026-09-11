"""一条命令求解四问、导出完整结果及可视化；参数来自所附模型，而非重新拟合。"""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
from drying_model import Config,Inputs,simulate
from export_results import export_run
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--out-dir',type=Path,default=ROOT/'results')
    p.add_argument('--figure-dir',type=Path,default=ROOT/'figures')
    p.add_argument('--n',type=int,default=320,help='radial intervals; default production grid=320')
    p.add_argument('--horizon-h',type=float,default=480.,help='safety horizon, not a prescribed drying time')
    p.add_argument('--no-excel',action='store_true',help='keep full CSV.GZ and raw diagnostics; skip XLSX packaging')
    p.add_argument('--no-plots',action='store_true')
    p.add_argument('--controls',action='store_true',help='append appendix-4 fixed-radius counterfactual')
    p.add_argument('--sensitivity',action='store_true',help='append 80-interval beta/Kp parameter study')
    p.add_argument('--grid-check',action='store_true',help='run 80/160/320 interval and time-refinement tests')
    p.add_argument('--gif',action='store_true',help='export an animated Q4 cross-section (requires Pillow)')
    args=p.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    inp=Inputs.load(args.data_dir)
    (args.out_dir/'input_audit.json').write_text(json.dumps(inp.audit(),ensure_ascii=False,indent=2),encoding='utf-8')
    summaries={}
    for case in (1,2,4):
        cfg=Config(case=case,n=args.n,shrink=case==4,stop_at_target=case!=1,
                   horizon_h=.5 if case==1 else args.horizon_h)
        print(f'Computing case {case}, {args.n+1} nodes...',flush=True)
        run=simulate(cfg,inp,progress=True)
        summaries[f'case{case}']=export_run(run,args.out_dir,ROOT/'data',excel=not args.no_excel)
        print(f'Completed case {case}; end={run.end_s/3600:.8f} h; target={run.reached}',flush=True)
        if case!=1 and not run.reached:
            print('WARNING: No threshold event. Report only the computed horizon, not a finite drying time.',file=sys.stderr)
        del run;gc.collect()
    if args.controls:
        run=simulate(Config(case=4,n=args.n,shrink=False,horizon_h=args.horizon_h),inp,progress=True)
        summaries['case4_fixed']=export_run(run,args.out_dir,ROOT/'data',excel=False,label='case4_fixed')
        del run;gc.collect()
    (args.out_dir/'all_summaries.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.sensitivity or args.grid_check:
        from validate import sensitivity_study,grid_study
        val=args.out_dir.parent/'validation'
        if args.sensitivity:sensitivity_study(inp,val)
        if args.grid_check:grid_study(inp,val)
    if not args.no_plots:
        from plot_results import make_plots
        make_plots(args.out_dir,args.figure_dir,args.data_dir,args.out_dir.parent/'validation',make_gif=args.gif)
    print('All requested computations and exports completed.',flush=True)

if __name__=='__main__':main()
