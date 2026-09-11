"""对BDF稠密解进行采样；无需重新积分即可导出四问全部表格。

输出节奏不是求解器的自适应步长。完整CSV.GZ和XLSX显示四位小数；
论文原始表、事件数据和可视化数组另行按全精度保存。收缩体外部位置
用空白/NaN表示，绝不用0表示。
"""
from __future__ import annotations
import csv,gzip,json
from pathlib import Path
import numpy as np
from drying_model import Run
from stream_excel import stream_xlsx


def output_times(end_s,step_s,start_s=0.):
    t=np.arange(start_s,end_s+1e-9,step_s,dtype=float)
    if not len(t) or abs(t[-1]-end_s)>1e-7:t=np.r_[t,end_s]
    else:t[-1]=end_s
    return t

def write_csv(path,headers,rows,rounded=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    op=gzip.open if path.suffix=='.gz' else open
    with op(path,'wt',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(headers)
        for row in rows:
            line=[]
            for j,v in enumerate(row):
                if not np.isfinite(v):line.append('')
                elif rounded and j>0:line.append(f'{v:.4f}')
                else:line.append(format(float(v),'.15g'))
            w.writerow(line)

def save_plot_arrays(run:Run,out:Path,label:str):
    early=min(14400.,run.end_s)
    t=np.unique(np.r_[np.arange(0.,early,5. if run.config.case==1 else 30.),
                         np.arange(early,run.end_s,120.),run.end_s])
    C,T=run.fields(t);d=run.diagnostics(t)
    np.savez_compressed(out/f'{label}_plots.npz',time_s=t,xi=run.x,C=C,T_C=T,
                        radius_m=run.radius_at(t),weights=run.weights,
                        config=json.dumps(run.config.__dict__),**{k:v for k,v in d.items() if k!='time_s'})
    write_csv(out/f'{label}_diagnostics.csv',list(d),zip(*d.values()))


def paper_tables(run:Run,out:Path):
    case=run.config.case
    radii=np.arange(0.,2.0001,.5)
    if case==1:
        times=np.array([100,300,600,900,1200,1500,1800.]);labels=[('table1_Q1_temperature',1),('table2_Q1_moisture',0)]
    elif case==2:
        times=np.arange(.5,3.001,.5)*3600;labels=[('table3_Q2_temperature',1),('table4_Q2_moisture',0)]
    else:
        times=output_times(run.end_s,21600.,21600.);labels=[('table6_Q4_moisture',0)]
    C,T=run.fields(times,radii);data=(C,T)
    for name,j in labels:
        vals=data[j];headers=['time_s']+[f'r_{r:.1f}_cm' for r in radii]
        if case==4:
            full=run.fields(times)[j]
            vals=np.c_[vals,full[:,-1],run.radius_at(times)*100]
            headers+=['surface','radius_cm']
        write_csv(out/f'{name}.csv',headers,np.c_[times,vals],rounded=True)
        write_csv(out/f'{name}_unrounded.csv',headers,np.c_[times,vals])
    if case==2:
        times=output_times(run.end_s,21600.,21600.)
        C,T=run.fields(times,radii)
        headers=['time_s']+[f'r_{r:.1f}_cm' for r in radii]
        write_csv(out/'table5_Q3_moisture.csv',headers,np.c_[times,C],rounded=True)
        write_csv(out/'table5_Q3_moisture_unrounded.csv',headers,np.c_[times,C])


def export_time_series(run:Run,out:Path,name:str,step_s:float,start_s:float,
                       template_dir:Path,with_temperature=True,shrink_extra=False,excel=True):
    times=output_times(run.end_s,step_s,start_s)
    radii=np.arange(21)/10.
    head=['time_s']+[f'{r:.1f}' for r in radii]
    if shrink_extra:head+=['surface','radius_cm']
    paths={'水分浓度':out/f'{name}_moisture.csv.gz'}
    if with_temperature:paths={'温度':out/f'{name}_temperature.csv.gz',**paths}
    # 每个文件只打开一次，每次稠密输出最多采样2000个时刻。
    import contextlib
    with contextlib.ExitStack() as stack:
        writers={key:csv.writer(stack.enter_context(gzip.open(path,'wt',encoding='utf-8-sig',newline='')))
                 for key,path in paths.items()}
        for w in writers.values():w.writerow(head)
        for begin in range(0,len(times),2000):
            tt=times[begin:begin+2000];C,T=run.fields(tt,radii)
            vals={'水分浓度':C,'温度':T}
            if shrink_extra:
                c,t=run.fields(tt);R=run.radius_at(tt)*100
                vals['水分浓度']=np.c_[C,c[:,-1],R];vals['温度']=np.c_[T,t[:,-1],R]
            for key,writer in writers.items():
                for ts,row in zip(tt,vals[key]):
                    writer.writerow([format(float(ts),'.15g')]+['' if not np.isfinite(v) else f'{v:.4f}' for v in row])
    if excel:
        kind='shrink' if shrink_extra else ('standard' if with_temperature else 'moisture')
        stream_xlsx(template_dir/f'{kind}_template.xlsx',paths,out/f'{name}.xlsx',len(times),allow_blank=shrink_extra)
    meta=dict(rows=len(times),start_s=float(times[0]),end_s=float(times[-1]),step_s=step_s,
              final_fractional_second_appended=bool(abs(times[-1]/step_s-round(times[-1]/step_s))>1e-8),
              columns=len(head),four_decimal_display=True,shrink_exterior_blank=shrink_extra,
              temperature_included=with_temperature,excel_exported=excel)
    (out/f'{name}_export.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    return meta


def export_run(run:Run,out_dir,template_dir,excel=True,label=None):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);template_dir=Path(template_dir)
    case=run.config.case;label=label or f'case{case}'
    met=run.metrics();met['energy_audit']=run.independent_energy_audit()
    (out/f'{label}_summary.json').write_text(json.dumps(met,ensure_ascii=False,indent=2),encoding='utf-8')
    save_plot_arrays(run,out,label)
    if label=='case4_fixed':return met
    paper_tables(run,out)
    if case==1:export_time_series(run,out,'result1',1.,1.,template_dir,excel=excel)
    elif case==2:
        export_time_series(run,out,'result2',1.,1.,template_dir,excel=excel)
        export_time_series(run,out,'result3',60.,0.,template_dir,with_temperature=False,excel=excel)
    else:export_time_series(run,out,'result4',60.,0.,template_dir,shrink_extra=True,excel=excel)
    return met
