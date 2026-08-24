import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 8.0,
    'axes.titlesize': 8.8,
    'axes.labelsize': 8.0,
    'xtick.labelsize': 6.9,
    'lines.solid_capstyle': 'round',
    'pdf.fonttype': 42,
})

h, d = 1.0, 0.42
xs, xi = 0.25, 2.62
blue   = '#2F6FB0'
green  = '#2A924A'
red    = '#C73A36'
purple = '#7553A6'
orange = '#D88421'
gray   = '#7D7D7D'
light  = '#BEBEBE'
black  = '#111111'

fig = plt.figure(figsize=(7.2, 2.48))
gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.06, 1.14], wspace=0.18)

# ---------------- (a) physical strip ----------------
ax = fig.add_subplot(gs[0,0])
ax.set_xlim(-0.05, 3.72)
ax.set_ylim(-0.25, 1.34)
ax.axis('off')
ax.text(0.00, 1.305, r'\textbf{(a) Billiard paths}', fontsize=8.8, ha='left', va='bottom')

ax.plot([0,3.24],[0,0], color=blue, lw=1.25)
ax.plot([0,3.24],[h,h], color=blue, lw=1.25)
ax.text(0.02, h+0.045, r'$y=h$', ha='left', va='bottom', fontsize=7.6)
ax.text(0.02, -0.045, r'$y=0$', ha='left', va='top', fontsize=7.6)

ax.scatter([xs],[0], marker='s', s=24, color=orange, zorder=5)
ax.text(xs, -0.145, 'source', ha='center', va='top', fontsize=7.4)
ax.scatter([xi],[d], s=28, color=blue, zorder=6)
ax.text(xi+0.06, d+0.02, 'ion', ha='left', va='center', fontsize=7.5)

ax.plot([xs,xi],[0,d], color=green, lw=1.45)
Y1 = 2*h-d
xb1 = xs + (xi-xs)*h/Y1
ax.plot([xs,xb1,xi],[0,h,d], color=red, lw=1.45)
Y2 = 2*h+d
xb2a = xs + (xi-xs)*h/Y2
xb2b = xs + (xi-xs)*(2*h)/Y2
ax.plot([xs,xb2a,xb2b,xi],[0,h,0,d], color=purple, lw=1.45)
ax.scatter([xb1],[h],s=15,color=red,zorder=6)
ax.scatter([xb2a,xb2b],[h,0],s=15,color=purple,zorder=6)

key_x0, key_x1 = 1.64, 1.98
for yy, col, lab in zip([1.205,1.125,1.045],[green,red,purple],['direct','1 bounce','2 bounces']):
    ax.plot([key_x0,key_x1],[yy,yy],color=col,lw=1.45,clip_on=False)
    ax.text(key_x1+0.08,yy,lab,va='center',ha='left',fontsize=7.2)

xd, xh = 3.02, 3.47
ax.annotate('',xy=(xd,d),xytext=(xd,0),arrowprops=dict(arrowstyle='<->',lw=0.7,color=black))
ax.text(xd+0.065,d/2,r'$d$',va='center',ha='left',fontsize=7.8)
ax.annotate('',xy=(xh,h),xytext=(xh,0),arrowprops=dict(arrowstyle='<->',lw=0.7,color=black))
ax.text(xh+0.065,h/2,r'$h$',va='center',ha='left',fontsize=7.8)

# ---------------- (b) unfolding ----------------
ax = fig.add_subplot(gs[0,1])
ax.set_xlim(-0.08, 3.50)
ax.set_ylim(-0.24, 2.72)
ax.axis('off')
ax.text(0.00, 2.66, r'\textbf{(b) Unfolding}', fontsize=8.8, ha='left', va='bottom')
ax.text(0.08, 2.42, r'$\alpha_n=|d-2nh|$', fontsize=9.0, ha='left', va='center')

for yy in [0,h,2*h]:
    ax.plot([0,3.34],[yy,yy],ls=(0,(4,3)),color=blue,lw=0.75,alpha=0.72)

sx=0.20
ax.scatter([sx],[0],marker='s',s=24,color=orange,zorder=5)
ax.text(sx,-0.14,'source',ha='center',va='top',fontsize=7.4)

alphas=[d,2*h-d,2*h+d]
cols=[green,red,purple]
xp=[2.12,2.53,2.94]
for a,c,x in zip(alphas,cols,xp):
    ax.plot([sx,x],[0,a],color=c,lw=1.45)
    ax.scatter([x],[a],s=27,color=c,zorder=5)

ax.text(2.24, d+0.02, r'$\alpha_0=d$', ha='left', va='center', fontsize=7.6)
ax.text(2.66, 2*h-d+0.01, r'$\alpha_1=2h-d$', ha='left', va='center', fontsize=7.6)
ax.text(2.88, 2*h+d-0.13, r'$\alpha_{-1}=2h+d$', ha='right', va='top', fontsize=7.6)

# ---------------- (c) depth spectrum ----------------
ax = fig.add_subplot(gs[0,2])
ax.set_xlim(0.0, 6.9)
ax.set_ylim(-0.02, 1.28)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.7)
ax.spines['bottom'].set_linewidth(0.7)
ax.tick_params(axis='y',left=False,labelleft=False)
ax.tick_params(axis='x',bottom=False,labelbottom=False)
ax.set_title(r'\textbf{(c) Return-depth spectrum}', loc='left', fontsize=8.8, pad=3)

nseq=np.array([0,1,-1,2,-2,3,-3])
a=np.abs(d-2*nseq*h)
k=0.34
w=np.exp(-k*a)
ord=np.argsort(a)
a,w=a[ord],w[ord]

for i,(aa,ww) in enumerate(zip(a,w)):
    col=[green,red,purple][i] if i<3 else gray
    stem_col=blue if i<3 else light
    ax.vlines(aa,0,ww,color=stem_col,lw=1.05,zorder=1)
    ax.scatter([aa],[ww],s=20,color=col,zorder=3)
xx=np.linspace(0.25,6.65,240)
ax.plot(xx,np.exp(-k*xx),color=light,lw=0.75,ls='--',zorder=0)

ax.set_xticks([])
ax.set_xlabel(r'unfolded depth $a$',labelpad=3)
ax.text(d+0.10, w[0]+0.01, r'$a=d$', ha='left', va='center', fontsize=7.0)
ax.text((2*h-d)+0.12, w[1]+0.01, r'$a=2h-d$', ha='left', va='center', fontsize=7.0)
ax.text((2*h+d)+0.14, w[2]-0.02, r'$a=2h+d$', ha='left', va='center', fontsize=7.0)

ax.text(0.06,0.92,r'$\mu_h=\sum_n\delta_{\alpha_n}$',transform=ax.transAxes,
        ha='left',va='top',fontsize=8.3)
ax.text(0.52,0.82,r'Laplace weight $e^{-ka}$',transform=ax.transAxes,
        ha='left',va='top',fontsize=7.4)

fig.subplots_adjust(left=0.012,right=0.995,bottom=0.16,top=0.965)
root='.'
fig.savefig(root+'/fig_unfolding_schematic.pdf',bbox_inches='tight',pad_inches=0.01)
fig.savefig(root+'/fig_unfolding_schematic.png',dpi=360,bbox_inches='tight',pad_inches=0.01)
