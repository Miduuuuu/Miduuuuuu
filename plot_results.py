"""可复现、有来源依据的四问数值解图表。

每张图单独保存，不强行套用统一配色。导出PNG和SVG，并额外生成
内容一致的矢量PDF图册。所有场量均来自NPZ绘图样本，而不是旧模型输出。
GIF是径向计算结果按轴对称重建的动画，不是单独的二维仿真。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parent
C_LABEL = r'Moisture $C$ (kg water / kg dry solid)'
T_LABEL = r'Temperature ($^\circ$C)'


def load_case(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def profile_at(d, time_s, field='C'):
    """仅对已经保存的可视化样本沿时间方向插值。"""
    return np.array([np.interp(time_s, d['time_s'], col) for col in d[field].T])


def physical_field(d, radius_cm, field='C'):
    """将材料坐标映射到固定距离；材料外部保持为NaN。"""
    result = np.full((len(d['time_s']), len(radius_cm)), np.nan)
    for i, R in enumerate(d['radius_m']):
        xi_query = np.asarray(radius_cm)/(100*R)
        inside = xi_query <= 1+1e-12
        result[i, inside] = np.interp(xi_query[inside], d['xi'], d[field][i])
    return result


def make_plots(results_dir=ROOT/'results', figures_dir=ROOT/'figures',
               data_dir=ROOT/'data', validation_dir=ROOT/'validation', make_gif=False):
    results_dir, figures_dir, data_dir, validation_dir = map(Path,
        (results_dir, figures_dir, data_dir, validation_dir))
    figures_dir.mkdir(parents=True, exist_ok=True)
    data = {k: load_case(results_dir/f'{k}_plots.npz') for k in ('case1','case2','case4')}
    if (results_dir/'case4_fixed_plots.npz').exists():
        data['case4_fixed'] = load_case(results_dir/'case4_fixed_plots.npz')
    summaries = json.loads((results_dir/'all_summaries.json').read_text(encoding='utf-8'))
    air = np.loadtxt(data_dir/'air.csv', delimiter=',', skiprows=1, encoding='utf-8-sig')
    rad = np.loadtxt(data_dir/'radius.csv', delimiter=',', skiprows=1, encoding='utf-8-sig')
    manifest=[]
    with PdfPages(figures_dir/'all_figures.pdf') as gallery:
        def save(fig, name, title_cn, note):
            fig.tight_layout(pad=1.4)
            fig.savefig(figures_dir/f'{name}.png', dpi=170)
            fig.savefig(figures_dir/f'{name}.svg')
            gallery.savefig(fig)
            manifest.append(dict(id=name, title=title_cn, note=note,
                                 png=name+'.png', svg=name+'.svg'))
            plt.close(fig)

        def chart(title, xlabel='Time (h)', ylabel=C_LABEL):
            fig, ax = plt.subplots(figsize=(8.6,5.1))
            ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
            ax.grid(True, alpha=.23)
            return fig, ax

        fig,ax=chart('Measured air temperature and the prescribed continuation',ylabel=T_LABEL)
        ax.plot(air[:,0]/3600, air[:,1],label='Measured / linear interpolation')
        tail=air[air[:,0]>=10800,1].mean()
        ax.plot([4,8],[tail,tail], '--',label='After 4 h: last-hour mean')
        ax.axvline(4,linestyle=':',linewidth=1);ax.legend()
        save(fig,'01_air_temperature','空气温度输入','0—4 h 为找回的记录；4 h 后为文中规定的末小时均值延拓。')

        fig,ax=chart('Measured air humidity ratio',ylabel='Humidity ratio (kg water / kg dry air)')
        ax.plot(air[:,0]/3600,air[:,2]);ax.set_xlim(0,4)
        save(fig,'02_air_humidity','空气含湿量输入','空气含湿量的分母是干空气质量，不是药材干物质量。')

        fig,ax=chart('Measured radius and the prescribed continuation',ylabel='Radius (cm)')
        ax.plot(rad[:,0]/3600,rad[:,1],label='Measured / linear interpolation')
        end=data['case4']['time_s'][-1]/3600
        ax.plot([72,end],[rad[-1,1],rad[-1,1]],'--',label='After 72 h: fixed final radius')
        ax.axvline(72,linestyle=':',linewidth=1);ax.legend()
        save(fig,'03_radius','半径及 72 h 后延拓','只有径向收缩，长度保持 0.25 m；72 h 后半径保持 1.198 cm 是额外延拓情景。')

        d=data['case1'];r=np.arange(0,2.001,.5)
        fig,ax=chart('Q1: radial temperature histories',xlabel='Time (s)',ylabel=T_LABEL)
        vals=physical_field(d,r,'T_C')
        for j,rv in enumerate(r):ax.plot(d['time_s'],vals[:,j],label=f'r = {rv:g} cm')
        ax.legend(ncol=2)
        save(fig,'04_Q1_temperature','第一问：不同半径处温度','计算时长为 1800 s，真实轴心和真实表面均为网格节点。')

        fig,ax=chart('Q1: radial moisture profiles',xlabel='Radius (cm)')
        for tt in (100,600,1200,1800):ax.plot(2*d['xi'],profile_at(d,tt),label=f't = {tt} s')
        ax.legend()
        save(fig,'05_Q1_moisture_profiles','第一问：径向含水率剖面','单位为干基 kg/kg，中心失水很慢而近表面梯度先建立。')

        fig,ax=chart('Q1: temperature in radius and time',xlabel='Time (s)',ylabel='Radius (cm)')
        im=ax.pcolormesh(d['time_s'],2*d['xi'],d['T_C'].T,shading='auto',rasterized=True)
        fig.colorbar(im,ax=ax,label=T_LABEL)
        save(fig,'06_Q1_temperature_map','第一问：温度时空图','横轴时间，纵轴距轴线距离，色标为摄氏温度。')

        d=data['case2'];sel=d['time_s']<=10800
        for field,label,name,titlecn in [('T_C',T_LABEL,'07_Q2_temperature','第二问：前 3 h 温度'),
                                         ('C',C_LABEL,'08_Q2_moisture','第二问：前 3 h 含水率')]:
            fig,ax=chart('Q2: first 3 hours with local dynamic properties',ylabel=label)
            vals=physical_field(d,r,field)
            for j,rv in enumerate(r):ax.plot(d['time_s'][sel]/3600,vals[sel,j],label=f'r = {rv:g} cm')
            ax.legend(ncol=2);ax.set_xlim(0,3)
            save(fig,name,titlecn,'从题给初态独立启动，所有位置的物性按当地含水率、温度更新。')

        for case,name,titlecn in [('case2','09_Q3_drying','第三问：全域最大含水率达标'),
                                  ('case4','10_Q4_drying','第四问：收缩模型达标')]:
            d=data[case];fig,ax=chart(('Q3' if case=='case2' else 'Q4')+': drying criterion over the whole material')
            for key,label,ls in [('C_max','Maximum','-'),('C_mean','Dry-mass-weighted mean','--'),('C_surface','Surface',':')]:
                ax.plot(d['time_s']/3600,d[key],linestyle=ls,label=label)
            ax.axhline(.15,linestyle='-.',linewidth=1,label='Target: 0.15 kg/kg')
            t=d['time_s'][-1]/3600;ax.axvline(t,linestyle=':',linewidth=1)
            ax.annotate(f'{t:.4f} h',xy=(t,.15),xytext=(-12,34),textcoords='offset points',ha='right',
                        arrowprops={'arrowstyle':'->'})
            ax.legend();ax.set_ylim(0,2.7)
            save(fig,name,titlecn,'达标判据是全域最大值，而不是均值或表面值；事件判断使用未舍入数值。')

        for case,name,titlecn in [('case2','11_Q3_moisture_map','第三问：含水率时空图'),
                                  ('case4','12_Q4_moving_boundary','第四问：物理坐标中的移动边界')]:
            d=data[case];rr=np.linspace(0,2,241);field=physical_field(d,rr)
            fig,ax=chart(('Q3' if case=='case2' else 'Q4')+': moisture on fixed physical radii',ylabel='Radius (cm)')
            im=ax.pcolormesh(d['time_s']/3600,rr,field.T,shading='auto',vmin=0,vmax=2.55,rasterized=True)
            ax.plot(d['time_s']/3600,d['radius_m']*100,linestyle='--',linewidth=1.5,label='Current surface')
            fig.colorbar(im,ax=ax,label=C_LABEL);ax.set_ylim(0,2.04);ax.legend(loc='lower left')
            save(fig,name,titlecn,'收缩后落在材料外的区域被掩膜为空白；不能把空白解读为含水率等于零。')

        d=data['case2'];fig=plt.figure(figsize=(8.8,5.8));ax=fig.add_subplot(111,projection='3d')
        ti=np.unique(np.linspace(0,len(d['time_s'])-1,100,dtype=int));ri=np.arange(0,len(d['xi']),4)
        xx,yy=np.meshgrid(d['time_s'][ti]/3600,2*d['xi'][ri],indexing='ij')
        ax.plot_surface(xx,yy,d['C'][np.ix_(ti,ri)],rstride=1,cstride=1,linewidth=0,antialiased=True)
        ax.set(title='Q3: computed moisture surface',xlabel='Time (h)',ylabel='Radius (cm)',zlabel='C (kg/kg)')
        ax.view_init(elev=26,azim=-125)
        save(fig,'13_Q3_surface_3D','第三问：三维曲面','三轴分别是时间、径向距离、含水率；不是三维空间 PDE。')

        d=data['case4'];tt=48*3600;R=np.interp(tt,d['time_s'],d['radius_m'])*100
        coords=np.linspace(-2.04,2.04,241);xx,yy=np.meshgrid(coords,coords);rr=np.hypot(xx,yy)
        cc=np.interp(np.minimum(rr/R,1),d['xi'],profile_at(d,tt));cc[rr>R]=np.nan
        fig,ax=chart('Q4: axisymmetric cross-section at 48 h',xlabel='x (cm)',ylabel='y (cm)')
        im=ax.pcolormesh(xx,yy,cc,shading='auto',vmin=0,vmax=2.55,rasterized=True)
        angles=np.linspace(0,2*np.pi,300);ax.plot(R*np.cos(angles),R*np.sin(angles),linewidth=1)
        ax.set_aspect('equal');ax.grid(False);fig.colorbar(im,ax=ax,label=C_LABEL)
        save(fig,'14_Q4_cross_section','第四问：48 h 轴对称截面','将一维径向结果按轴对称展开，不能据此声称已求解二维非均匀受热。')

        fig,ax=chart('Drying-time comparison and the shrinkage-only control',ylabel='Drying time (h)',xlabel='Model')
        entries=[('Q3\nAppendix 3','case2'),('Q4\nAppendix 4 + shrinkage','case4')]
        if 'case4_fixed' in summaries:entries.append(('Control\nAppendix 4, fixed radius','case4_fixed'))
        labels=[a for a,b in entries];times=[summaries[b]['drying_h'] for a,b in entries]
        bars=ax.bar(labels,times)
        ax.bar_label(bars,labels=[f'{v:.4f} h' for v in times],padding=5)
        ax.set_ylim(0,max(times)*1.15)
        save(fig,'15_drying_time_comparison','达标时间及收缩单因素对照','Q3 与 Q4 的物性不同。只有 Q4 与“附录4、固定半径”之差可在本模型内用于隔离收缩效应。')

        fig,ax=chart('Surface temperature, air temperature, and air dew point',ylabel=T_LABEL)
        d=data['case2'];sel=d['time_s']<=14400
        for key,label,ls in [('T_air_C','Air','-'),('T_surface_C','Q2 surface','--'),('T_dew_C','Air dew point',':')]:
            ax.plot(d['time_s'][sel]/3600,d[key][sel],linestyle=ls,label=label)
        ax.legend();ax.set_xlim(0,4)
        save(fig,'16_surface_temperature','表面、空气温度与露点','蒸发冷却由同一带符号通量和潜热自动产生；未强制修改温度。')

        fig,ax=chart('Q2: surface heat-flux terms during the first 4 hours',ylabel=r'Heat flux (W / m$^2$)')
        d=data['case2'];sel=d['time_s']<=14400;t=d['time_s'][sel]/3600;Ts=d['T_surface_C'][sel];j=d['j_e_kg_m2_s'][sel]
        qc=25*(d['T_air_C'][sel]-Ts);ql=(2.501e6-2361*Ts)*j
        ax.plot(t,qc,label='Convection into the surface');ax.plot(t,ql,'--',label='Latent heat carried out')
        ax.plot(t,qc-ql,':',label='Net conductive heat into the material');ax.axhline(0,linewidth=.8)
        ax.legend();ax.set_xlim(0,4)
        save(fig,'17_surface_heat_flux','对流供热、汽化耗热与净导热','全部使用 W/m²；导热净输入为 h(Tair−Ts)−Lv(Ts)je。')

        fig,ax=chart('Water conservation residual (normalized by initial water)',ylabel='Relative residual')
        for key,label in [('case1','Q1'),('case2','Q2 / Q3'),('case4','Q4')]:
            d=data[key];v=d['mass_residual_kg']/summaries[key]['initial_water_kg']
            ax.plot(d['time_s']/3600,v,label=label)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.legend()
        save(fig,'18_mass_balance','总水分收支误差','离散内部通量成对相消。该误差衡量数值守恒，不是模型预测不确定度或实验拟合误差。')

        path=validation_dir/'convergence.json'
        if path.exists():
            rows=json.loads(path.read_text(encoding='utf-8'))
            fig,ax=chart('Radial grid convergence of the drying time',xlabel='Radial intervals',ylabel='Difference from 320 intervals (s)')
            for case,label in [(2,'Q3'),(4,'Q4')]:
                r=[row for row in rows if row['study']=='space' and row['case']==case]
                r=sorted(r,key=lambda v:v['n']);base=r[-1]['drying_h']
                ax.plot([v['n'] for v in r],[(v['drying_h']-base)*3600 for v in r],'o-',label=label)
            ax.legend();ax.set_xticks([80,160,320])
            save(fig,'19_grid_convergence','网格加密检验','以 320 个径向间隔结果为参考；差值的单位是秒，不是小时。')
        path=validation_dir/'sensitivity.json'
        if path.exists():
            rows=json.loads(path.read_text(encoding='utf-8'))
            for parameter,xlabel,name,titlecn in [('beta',r'$\beta/\beta_0$','20_beta_sensitivity','水活度参数敏感性'),
                                                 ('Kp',r'$K_p/K_{p,0}$','21_Kp_sensitivity','外部传质尺度敏感性')]:
                fig,ax=chart('Parameter sensitivity of drying time (80-interval grid)',xlabel=xlabel,ylabel='Change from same-grid baseline (%)')
                for case,label in [(2,'Q3'),(4,'Q4')]:
                    rr=[v for v in rows if v['case']==case and v['parameter'] in ('baseline',parameter)]
                    key='beta_factor' if parameter=='beta' else 'Kp_factor';rr.sort(key=lambda v:v[key])
                    ax.plot([v[key] for v in rr],[v['change_pct'] for v in rr],'o-',label=label)
                if parameter=='Kp':ax.set_xscale('log');ax.set_xticks([.3,1,3],['0.3','1','3'])
                ax.axhline(0,linewidth=.8);ax.legend()
                save(fig,name,titlecn,'一次只改变一个闭合参数；百分比使用相同 80 网格基线。扰动情景不是统计置信区间。')
    (figures_dir/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    md=['# 图件索引','所有图件另有 PNG、SVG；all_figures.pdf 汇集相同图件。','']
    for v in manifest:md.extend([f"## {v['id']}：{v['title']}",v['note'],''])
    (figures_dir/'图件说明.md').write_text('\n'.join(md),encoding='utf-8')
    if make_gif:make_shrink_gif(results_dir,figures_dir)
    return manifest


def make_shrink_gif(results_dir=ROOT/'results',figures_dir=ROOT/'figures',nframes=55):
    d=load_case(Path(results_dir)/'case4_plots.npz');out=Path(figures_dir);out.mkdir(parents=True,exist_ok=True)
    coords=np.linspace(-2.04,2.04,165);xx,yy=np.meshgrid(coords,coords);rr=np.hypot(xx,yy)
    times=np.linspace(0,d['time_s'][-1],nframes);theta=np.linspace(0,2*np.pi,220)
    fig,ax=plt.subplots(figsize=(6.8,5.7))
    artist=ax.imshow(np.full_like(rr,np.nan),origin='lower',extent=[coords[0],coords[-1],coords[0],coords[-1]],vmin=0,vmax=2.55)
    line,=ax.plot([],[],linewidth=1);title=ax.set_title('Q4: radial solution shown as an axisymmetric section')
    ax.set(xlabel='x (cm)',ylabel='y (cm)',xlim=(-2.04,2.04),ylim=(-2.04,2.04))
    ax.set_aspect('equal');fig.colorbar(artist,ax=ax,label=C_LABEL);fig.tight_layout()
    def update(idx):
        t=times[idx];R=np.interp(t,d['time_s'],d['radius_m'])*100
        field=np.interp(np.minimum(rr/R,1),d['xi'],profile_at(d,t));field[rr>R]=np.nan
        artist.set_data(field);line.set_data(R*np.cos(theta),R*np.sin(theta))
        title.set_text(f'Q4 axisymmetric reconstruction | {t/3600:.2f} h | R = {R:.3f} cm')
        return artist,line,title
    animation=FuncAnimation(fig,update,frames=nframes,interval=150,blit=False)
    animation.save(out/'Q4_shrinkage.gif',writer=PillowWriter(fps=7),dpi=95)
    plt.close(fig)
    return out/'Q4_shrinkage.gif'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir',type=Path,default=ROOT/'results');p.add_argument('--figures-dir',type=Path,default=ROOT/'figures')
    p.add_argument('--data-dir',type=Path,default=ROOT/'data');p.add_argument('--validation-dir',type=Path,default=ROOT/'validation')
    p.add_argument('--gif',action='store_true');args=p.parse_args()
    manifest=make_plots(args.results_dir,args.figures_dir,args.data_dir,args.validation_dir,args.gif)
    print(f'Exported {len(manifest)} separate figures with PNG/SVG and the PDF gallery.')
