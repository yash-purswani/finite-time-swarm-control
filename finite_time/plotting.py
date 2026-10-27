"""Reusable print figures, separate from simulation; normalized units in report."""
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.lines import Line2D
import numpy as np
from .style import use_paper_style,COL2_W,COL_W

COLORS={0.: '#555555',.5:'#0072B2',2.:'#009E73',2.5:'#2CA02C',5.:'#E69F00',10.:'#CC79A7',25.:'#D55E00'}


def save(fig,folder,name):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(folder/f'{name}.png',dpi=300,bbox_inches='tight')
    plt.close(fig)


def positive(ax,t,y,**kw):
    # Mask numerical zeros; no logarithm is taken and data are never rewritten.
    ax.plot(t,np.ma.masked_where(y<=0,y),**kw)


def plot_morphing(ar,c,folder):
    use_paper_style()
    fig,axes=plt.subplots(1,4,figsize=(COL2_W,2.65),sharey=True)
    colors=plt.cm.tab10(np.arange(c['n']))
    names=['Ring','Wedge','Star','Arrow']
    t=ar['t'];X=ar['X'];q=ar['q'];r=q.mean(axis=1)
    snapshot_indices=[int(np.argmin(abs(t-tq))) for tq in c['snapshot_times']]
    shown=np.concatenate([np.concatenate([q[index],X[np.searchsorted(t,max(0,tq-1.5)):index+1].reshape(-1,2)])
                          for index,tq in zip(snapshot_indices,c['snapshot_times'])])
    y_limits=(shown[:,1].min()-.4,shown[:,1].max()+.4)
    for ax,tq,name in zip(axes,c['snapshot_times'],names):
        index=int(np.argmin(abs(t-tq)));start=np.searchsorted(t,max(0,tq-1.5))
        from .formations import hexagon,wedge,star,arrow
        from scipy.spatial import ConvexHull
        L=c['radius']
        template={'Ring':lambda:hexagon(c['n'],L),'Wedge':lambda:wedge(c['n'],.6*L),
                  'Star':lambda:star(c['n'],1.1*L),'Arrow':lambda:arrow(c['n'],L)}[name]()
        if name=='Wedge':template=template[ConvexHull(template).vertices]
        outline=np.vstack([template,template[0]])+r[index]
        ax.plot(outline[:,0],outline[:,1],color='#aaa',lw=.6,alpha=.6,zorder=0)
        for i,col in enumerate(colors):
            ax.plot(X[start:index+1,i,0],X[start:index+1,i,1],color=col,lw=.65,alpha=.7)
            ax.scatter(*q[index,i],s=38,facecolors='none',edgecolors=[col],lw=.8,zorder=4)
            ax.scatter(*X[index,i],s=12,color=col,zorder=5)
            ax.annotate(str(i+1),q[index,i],xytext=(3,3),textcoords='offset points',fontsize=5,color=col)
        ax.plot(r[start:index+1,0],r[start:index+1,1],'k--',lw=.8)
        ax.plot(*r[index],'+',color='k',ms=6)
        # Set limits explicitly: equal aspect with shared y and adjustable
        # datalim can silently crop stations on the outer edges.
        local=np.concatenate([q[index],X[start:index+1].reshape(-1,2)])
        ax.set_xlim(local[:,0].min()-.4,local[:,0].max()+.4)
        ax.set_ylim(*y_limits)
        ax.set_aspect('equal',adjustable='box')
        ax.set_title(f'{name}, $t={tq:g}$')
        ax.set_xlabel('$x$ (normalized)')
    axes[0].set_ylabel('$y$ (normalized)')
    fig.legend(handles=[Line2D([],[],marker='o',color='k',ls='',ms=3,label='agents'),
                        Line2D([],[],marker='o',markerfacecolor='none',color='k',ls='',ms=5,label='assigned stations'),
                        Line2D([],[],color='k',ls='--',label='reference trajectory')],
               loc='outside upper center' if matplotlib.__version__>='3.7' else 'upper center',ncol=3,
               bbox_to_anchor=(.5,1.11),frameon=False)
    save(fig,folder,'formation_snapshots')


def plot_errors(runs,folder):
    use_paper_style();fig,axes=plt.subplots(1,3,figsize=(COL2_W,2.55))
    floor = 1e-4
    for m,ar,me in runs:
        color=COLORS.get(m, '#009E73');t=ar['t']
        positive(axes[0],t,np.ma.masked_where(ar['delta_norm']<floor,ar['delta_norm']),color=color,label=f'$m={m:g}$')
        positive(axes[1],t,np.ma.masked_where(ar['eta_norm']<floor,ar['eta_norm']),color=color)
        positive(axes[1],t,np.ma.masked_where(ar['eta_bound']<floor,ar['eta_bound']),color=color,ls='--',lw=.7)
        axes[2].plot(t,ar['u_norm'],color=color)
    ar=runs[0][1];deadline=runs[0][2]['T_bar']
    positive(axes[0],ar['t'],np.ma.masked_where(ar['R_bar']<floor,ar['R_bar']),color='k',ls='--',lw=1.1,label=r'$\overline{R}$')
    for ax in axes[:2]:
        ax.set_yscale('log');ax.set_ylim(floor,max(1.0,1.5*runs[0][1]['R_bar'][0]))
        ax.axvline(deadline,color='k',ls=':',lw=.8)
    axes[0].text(deadline-.04,2e-4,r'$\overline{T}$',ha='right',fontsize=7)
    axes[0].legend(loc='upper right',ncol=2,fontsize=6)
    axes[1].legend(handles=[Line2D([],[],color='k',label='measured'),
        Line2D([],[],color='k',ls='--',label=r'bound: integrated $\lambda_2(L_w)$')],loc='upper right',fontsize=5.8)
    for ax in axes:
        ax.set_xlabel('$t$ (normalized)');ax.set_xlim(0,ar['t'][-1])
    for ax in axes[:2]:
        ax.set_xlim(0,min(ar['t'][-1],1.15*deadline))
    axes[0].set_ylabel(r'$\|\delta\|_2$');axes[1].set_ylabel(r'$\|\eta\|_2$')
    axes[2].set_ylabel(r'$\|u\|_2$ (normalized)')
    inset=axes[2].inset_axes([.43,.48,.54,.42])
    initial_commands=[]
    for m,ar,me in runs:
        mask=ar['t']<=.12
        inset.plot(ar['t'][mask],ar['u_norm'][mask],color=COLORS.get(m, '#009E73'),lw=.8)
        initial_commands.extend(ar['u_norm'][mask])
    lower,upper=min(initial_commands),max(initial_commands)
    pad=max((upper-lower)*.08,1e-3)
    inset.set_xlim(0,min(.12,ar['t'][-1]));inset.set_ylim(lower-pad,upper+pad)
    inset.tick_params(labelsize=5);inset.set_title('Initial command',fontsize=6)
    for ax,label in zip(axes,['(a) Station tracking','(b) Disagreement','(c) Physical command']):
        ax.set_title(label)
    save(fig,folder,'error_disagreement_control')


def plot_star_maneuver(ar,folder):
    """Station/physical snapshots with fixed agent assignments along the path."""
    use_paper_style()
    fig,ax=plt.subplots(figsize=(COL2_W,3.3))
    t=ar['t'];q=ar['q'];X=ar['X'];r=q.mean(axis=1)
    colors=plt.cm.tab10(np.arange(q.shape[1]));closed=np.r_[np.arange(q.shape[1]),0]
    ax.plot(r[:,0],r[:,1],'k--',lw=.9,label='reference path')
    for tau in np.linspace(t[0],t[-1],5):
        i=np.argmin(abs(t-tau))
        ax.plot(q[i,closed,0],q[i,closed,1],color='#888888',lw=.75)
        ax.scatter(q[i,:,0],q[i,:,1],s=23,facecolors='none',edgecolors=colors,lw=.7)
        ax.scatter(X[i,:,0],X[i,:,1],s=6,c=colors,zorder=3)
        ax.annotate(f'$t={t[i]:g}$',r[i],xytext=(3,-15),textcoords='offset points',fontsize=6)
    ax.plot([],[],'o',mfc='none',mec='k',ms=4,label='stations')
    ax.plot([],[],'o',color='k',ms=2,label='agents')
    ax.set_aspect('equal',adjustable='box');ax.set_xlabel('$x$ (normalized)');ax.set_ylabel('$y$ (normalized)')
    ax.legend(loc='upper right',fontsize=6)
    save(fig,folder,'star_maneuver')


def plot_two_agent(runs,folder):
    use_paper_style();fig,axes=plt.subplots(1,2,figsize=(COL2_W,2.35))
    for m,ar,me in runs:
        col=COLORS[m]
        axes[0].plot(ar['t'],ar['d_x'],color=col,label=f'$m={m:g}$')
        axes[1].plot(ar['t'],ar['delta'][:,0,0],color=col,label=f'$m={m:g}$')
        axes[1].plot(ar['t'][::30],ar['exact_error'][::30],'o',mfc='none',mec=col,ms=3)
    axes[0].axhline(1,color='k',ls='--',label='$d_{safe}$')
    axes[0].set_xlim(0,.8);axes[0].set_ylim(0,2.3)
    axes[0].set_ylabel('Actual separation (normalized)')
    axes[1].set_ylabel('$e$ (normalized)')
    axes[1].plot([],[],'o',mfc='none',mec='k',label='closed form')
    for ax in axes:ax.set_xlabel('$t$ (normalized)');ax.legend()
    save(fig,folder,'analytical_two_agent')


def plot_safety(runs,folder,d_safe=1.,R_c=60.,name='safety_connectivity',graph_name='graph_history'):
    use_paper_style();fig,axes=plt.subplots(2,3,figsize=(COL2_W,4.5),sharex=True)
    for row,(m,ar,me,cert) in enumerate(runs):
        t=ar['t'];color=COLORS.get(m,'#D55E00');axs=axes[row]
        label='certified tube' if cert['certified'] else 'graph-independent tube'
        axs[0].plot(t,ar['d_x'],color=color,label='$d_x$')
        axs[0].plot(t,ar['d_q']-ar['rho_H'],'k--',lw=.9,label=label+' lower')
        axs[0].axhline(d_safe,color='#888888',ls=':',label='$d_{safe}$')
        axs[1].plot(t,ar['ell_x_H'],color=color,label=r'$\ell_{x,H}$')
        axs[1].plot(t,ar['ell_H']+ar['rho_H'],'k--',lw=.9,label=label+' upper')
        axs[1].axhline(R_c,color='#888888',ls=':',label='$R_c$')
        axs[2].plot(t,ar['s'],color='#0072B2',label='$s$')
        axs[2].plot(t,ar['rho_H'],color=color,ls='--',label=r'$\rho_H$' if cert['certified'] else r'$\rho_{ind}$')
        for ax in axs:
            ax.axvline(me['T_bar'],color='#aaa',ls=':',lw=.6)
            ax.legend(fontsize=5.8,loc='upper right')
            ax.set_xlim(t[0],t[-1])
        status='certified' if cert['certified'] else 'uncertified'
        axs[0].set_ylabel(f'$m={m:g}$ ({status})\nDistance (normalized)')
        axs[1].set_ylabel('Length (normalized)');axs[2].set_ylabel('Margin (normalized)')
    for ax,label in zip(axes[0],['(a) Minimum separation','(b) Backbone length','(c) Available margin and distortion']):
        ax.set_title(label)
    for ax in axes[1]:ax.set_xlabel('$t$ (normalized)')
    save(fig,folder,name)
    # Auxiliary graph evidence: complete H is a legitimate connected spanning
    # backbone. Its density is explicit; no sparse-network result is implied.
    fig,axes=plt.subplots(1,2,figsize=(COL2_W,2.3))
    for m,ar,me,cert in runs:
        col=COLORS.get(m,'#D55E00')
        axes[0].plot(ar['t'],ar['lambda2'],color=col,label=f'$m={m:g}$')
        axes[1].plot(ar['t'],ar['active_edges'].sum(axis=(1,2))/2,color=col,label=f'$m={m:g}$')
    axes[0].axhline(runs[0][3]['w_lower']*runs[0][3]['lambda2_H'],color='k',ls='--',label='backbone spectral floor')
    axes[0].set_ylabel(r'$\lambda_2(L_w)$');axes[1].set_ylabel('Active communication edges')
    for ax in axes:ax.set_xlabel('$t$ (normalized)');ax.legend(fontsize=6)
    save(fig,folder,graph_name)


def _component_count(adjacency):
    """Number of connected components in each sampled undirected graph."""
    counts=[]
    for graph in adjacency:
        n=len(graph);seen=set();count=0
        for root in range(n):
            if root in seen:continue
            count+=1;stack=[root];seen.add(root)
            while stack:
                i=stack.pop()
                for j in np.flatnonzero(graph[i]):
                    j=int(j)
                    if j not in seen:seen.add(j);stack.append(j)
        counts.append(count)
    return np.asarray(counts)


def _plot_star_2x3_comparison(cases, labels, out_paths, transient_end=2.5, d_safe=1.0, r_c=6.0):
    """Plot 2x3 grid: Row 1 = Certified Safe, Row 2 = Unsafe Mid-Transient Collision."""
    use_paper_style()
    fig, axes = plt.subplots(2, 3, figsize=(COL2_W, 4.3), sharex=True, layout='none')
    
    row_colors = ['#009E73', '#e74c3c']
    row_titles = [r'Row 1: Safe ($m=2.5$)', r'Row 2: Collision ($m=0.0$)']
    
    for row, case in enumerate(cases):
        t = case['t']
        mask = t <= transient_end
        tz = t[mask]
        dx = case['d_x'][mask]
        ellx = case['ell_x'][mask]
        dq = case['dq'][mask] if 'dq' in case else (case['d_q'][mask] if 'd_q' in case else None)
        ellq = case['ell_q'][mask] if 'ell_q' in case else (case['ell_H'][mask] if 'ell_H' in case else None)
        s = case['s'][mask]
        rho = case['rho_H'][mask]
        color = row_colors[row % len(row_colors)]
        certified = bool(case.get('certified', False))
        
        # Col 0: Minimum pair separation dx(t)
        ax0 = axes[row, 0]
        ax0.plot(tz, dx, color=color, lw=1.6, label=r'$d_x(t)$ (observed)')
        if dq is not None:
            ax0.plot(tz, dq - rho, color='k', ls='--', lw=1.1, label=r'$d_q(t) - \rho_{\mathcal{H}}(t)$')
        ax0.axhline(d_safe, color='#cc0000', ls=':', lw=1.4, label=f'$d_{{\\mathrm{{safe}}}} = {d_safe:g}$ m')
        if row == 1:
            ax0.fill_between(tz, dx, d_safe, where=(dx < d_safe), color='#ff4d4d', alpha=0.40,
                             label='collision ($d_x < d_{\\mathrm{safe}}$)', zorder=2)
        ax0.set_ylabel(f"{row_titles[row]}\nDistance [m]")
        ax0.set_ylim(0.85, 1.65)
        ax0.legend(loc='upper right', fontsize=5.8)
        
        # Col 1: Maximum pair separation ell_x(t)
        ax1 = axes[row, 1]
        ax1.plot(tz, ellx, color=color, lw=1.6, label=r'$\ell_x(t)$ (observed)')
        if ellq is not None:
            ax1.plot(tz, ellq + rho, color='k', ls='--', lw=1.1, label=r'$\ell_q(t) + \rho_{\mathcal{H}}(t)$')
        ax1.axhline(r_c, color='#cc0000', ls=':', lw=1.4, label=f'$R_c = {r_c:g}$ m')
        ax1.set_ylabel("Length [m]")
        ax1.set_ylim(3.5, 6.3)
        ax1.legend(loc='upper right', fontsize=5.8)
        
        # Col 2: Margin s(t) vs Distortion rho_H(t)
        ax2 = axes[row, 2]
        ax2.plot(tz, s, color='#0072B2', lw=1.8, label=r'$s(t) = d_q - d_{\mathrm{safe}}$')
        ax2.plot(tz, rho, color=color, ls='--', lw=1.4, label=r'$\rho_{\mathcal{H}}(t)$')
        if certified or np.all(s >= rho):
            ax2.fill_between(tz, rho, s, where=(s >= rho), color='#2ecc71', alpha=0.25,
                             label=r'certified ($s \geq \rho_{\mathcal{H}}$)')
        else:
            ax2.fill_between(tz, s, rho, where=(rho > s), color='#f39c12', alpha=0.35,
                             label=r'uncertified ($s < \rho_{\mathcal{H}}$)')
            ax2.fill_between(tz, rho, s, where=(s >= rho), color='#2ecc71', alpha=0.15)
        ax2.set_ylabel("Margin [m]")
        ax2.set_ylim(-0.05, 0.65)
        ax2.legend(loc='upper right', fontsize=5.8)
        
        for ax in axes[row]:
            ax.set_xlim(0, transient_end)
            t_bar = case.get('T_bar_univ', case.get('T_bar', None))
            if t_bar is not None and t_bar <= transient_end:
                ax.axvline(t_bar, color='#888888', ls=':', lw=0.7)
    
    axes[0, 0].set_title(r'(a) Minimum pairwise distance $d_x(t)$')
    axes[0, 1].set_title(r'(b) Maximum pairwise distance $\ell_x(t)$')
    axes[0, 2].set_title(r'(c) Geometric margin $s(t)$ vs. tube $\rho_{\mathcal{H}}(t)$')
    
    for ax in axes[1]:
        ax.set_xlabel(r'Time $t$ [s]')
        
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.11, top=0.92, hspace=0.25, wspace=0.28)
    for p in out_paths:
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches='tight', dpi=300)
    plt.close(fig)


_plot_star_3x3_comparison = _plot_star_2x3_comparison


def plot_experiment3_three_cases(cases,folder):
    """Compare safety/connectivity tubes and margins for three scenarios."""
    use_paper_style()
    fig,axes=plt.subplots(3,3,figsize=(COL2_W,6.1),sharex=False,layout='none')
    for row,case in enumerate(cases):
        ar=case['arrays'];meta=case['meta'];t=ar['t'];color=case.get('color','#0072B2')
        certified=bool(meta.get('certified',False))
        tube_label='certified tube' if certified else 'graph-independent tube'
        for ax in axes[row]:ax.set_xlim(t[0],t[-1])
        axes[row,0].plot(t,ar['d_x'],color=color,lw=1,label=r'$d_x$')
        axes[row,0].plot(t,ar['d_q']-ar['rho_H'],color='k',ls='--',lw=.9,
                         label=f'{tube_label} lower')
        axes[row,0].axhline(meta['d_safe'],color='#888888',ls=':',lw=.8,label=r'$d_{safe}$')
        axes[row,0].set_ylabel('Distance (normalized)')
        axes[row,1].plot(t,ar['ell_x_H'],color=color,lw=1,label=r'$\ell_{x,H}$')
        axes[row,1].plot(t,ar['ell_H']+ar['rho_H'],color='k',ls='--',lw=.9,
                         label=f'{tube_label} upper')
        axes[row,1].axhline(meta['R_c'],color='#888888',ls=':',lw=.8,label=r'$R_c$')
        axes[row,1].set_ylabel('Length (normalized)')
        axes[row,2].plot(t,ar['s'],color='#0072B2',lw=1,label=r'$s$')
        axes[row,2].plot(t,ar['rho_H'],color=color,ls='--',lw=1,
                         label=r'$\rho_H$' if certified else r'$\rho_{ind}$')
        axes[row,2].set_ylabel('Margin (normalized)')
        event=meta.get('event_time')
        if event is not None:
            for ax in axes[row]:ax.axvline(event,color='#D55E00',ls='--',lw=.8)
        axes[row,0].text(.025,.04,case.get('short_label',case['label']),
                         transform=axes[row,0].transAxes,ha='left',va='bottom',
                         fontsize=5.8,fontweight='bold',color=color,
                         bbox=dict(facecolor='white',edgecolor='none',alpha=.78,pad=.8))
        if row==0:
            axes[row,0].set_title('Minimum pair separation')
            axes[row,1].set_title('Longest backbone edge')
            axes[row,2].set_title('Available margin and distortion')
        for ax in axes[row]:
            ax.axvline(meta.get('T_bar',0),color='#aaa',ls=':',lw=.6)
            ax.legend(fontsize=5.8,loc='upper right')
    for ax in axes[-1]:ax.set_xlabel('$t$ (normalized)')
    fig.subplots_adjust(left=.09,right=.99,bottom=.09,top=.92,hspace=.34,wspace=.30)
    save(fig,folder,'experiment3_three_case_comparison')


def plot_map(kappas,ms,classes,rows,folder,boundary=None):
    use_paper_style()
    labels=['Category A: Certified safe by Corollary 2','Category B: Uncertified, observed safe',
            'Category C: Observed transient collision','Category D: Infeasible geometry','VALIDATION FAILURE']
    mapping={key:i for i,key in enumerate(['A','B','C','D','VALIDATION_FAILURE'])}
    z=np.vectorize(mapping.get)(classes)
    colors=['#009E73','#56B4E9','#D55E00','#BDBDBD','#CC00CC']
    fig,ax=plt.subplots(figsize=(COL_W,2.85))
    cmap=ListedColormap(colors);norm=BoundaryNorm(np.arange(-.5,5.5),5)
    ax.pcolormesh(kappas,ms,z,cmap=cmap,norm=norm,shading='nearest',rasterized=False)
    if boundary is None:
        boundary=(kappas,np.array([r['m_required'] if r['gamma_req'] is not None and r['gamma_req']>=1 else np.nan
                                 for r in rows[:len(kappas)]],dtype=float))
    boundary_k,boundary_m=boundary
    ax.plot(boundary_k,boundary_m,'k--',lw=1.2,label=r'Design boundary $m_{\mathrm{req}}(\kappa)$')
    ax.scatter(np.tile(kappas,len(ms)),np.repeat(ms,len(kappas)),s=6,c='k',alpha=.65)
    ax.set_title(f'{len(ms)} x {len(kappas)} sampled configurations',fontsize=7)
    ax.set_ylim(ms[0],ms[-1]);ax.set_xlim(kappas[0],kappas[-1])
    ax.set_xlabel(r'Maneuver compression rate $\kappa$');ax.set_ylabel(r'Coupling gain $m$')
    handles=[Line2D([],[],marker='s',ls='',color=colors[i],label=labels[i])
             for i,key in enumerate(['A','B','C','D','VALIDATION_FAILURE']) if np.any(classes==key)]
    if np.any(np.isfinite(boundary_m)):
        handles += [Line2D([],[],ls='--',color='k',label=r'Design boundary $m_{\mathrm{req}}(\kappa)$')]
    ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.25),
              fontsize=5.8,frameon=False)
    ax.grid(False)
    save(fig,folder,'parameter_map')
    peak=np.array([r['peak_control'] for r in rows]).reshape(len(ms),len(kappas))
    fig,ax=plt.subplots(figsize=(COL_W,2.6))
    image=ax.pcolormesh(kappas,ms,peak,cmap='viridis',shading='nearest',rasterized=False)
    ax.plot(boundary_k,boundary_m,'w--',lw=.9)
    ax.scatter(np.tile(kappas,len(ms)),np.repeat(ms,len(kappas)),s=5,c='w',alpha=.6)
    ax.set_xlabel(r'Maneuver compression rate $\kappa$');ax.set_ylabel(r'Coupling gain $m$')
    fig.colorbar(image,ax=ax,label=r'Peak $\|u\|_2$')
    ax.grid(False);save(fig,folder,'parameter_control')
