from __future__ import annotations
import math, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import zeta, i0e

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import billiard_anomalous_heating as bah
import screened_bem_billiards as sb
import many_ion_quantum_noise as mi
import curved_multi_ion_noise as cm

report=[]
def rec(label, value): report.append(f"{label}: {value}")
def require(cond,msg):
    if not cond: raise AssertionError(msg)

# A. Original/slab physics checks.
gy0=bah.G_perp(1,None); gx0=bah.G_tangent(1,None)
require(abs(gy0-3/(16*math.pi))<2e-10,'open normal prefactor')
require(abs(gx0-3/(32*math.pi))<2e-10,'open tangential prefactor')
z3=float(zeta(3,1)); eta3=.75*z3
ry=bah.G_perp(1,2)/gy0; rx=bah.G_tangent(1,2)/gx0
require(abs(ry-z3)<2e-9,'zeta3')
require(abs(rx-eta3)<2e-9,'eta3')
require(abs(bah.G_perp(1,2)/bah.G_tangent(1,2)-8/3)<2e-9,'8/3 anisotropy')
rec('open Gy/Gx prefactors',f'{gy0:.12g} / {gx0:.12g}')
rec('h/d=2 normal/tangential ratios',f'{ry:.12f} / {rx:.12f}')

for h in [1.05,1.1,1.3,1.5,2,3,6,12]:
    for k in np.logspace(-5,2,180):
        ay=bah.kperp(k,1,h)/bah.kperp(k,1,None)
        ax=bah.potential_ratio(k,1,h)/bah.potential_ratio(k,1,None)
        require(ay>1-2e-12,'normal multiplier inequality')
        require(0<ax<1+2e-12,'tangential multiplier inequality')
rec('pointwise polarization inequalities','PASS')

# sharp coefficient
def cross_C(h):
    a=h-1
    def integrand(k):
        if k==0:return 0.0
        frac=(math.exp(-2*a*k)+math.exp(-2*h*k))/(-math.expm1(-2*h*k))
        return 2*math.pi*k*k*i0e(k)*frac
    return quad(integrand,0,np.inf,epsabs=2e-10,epsrel=2e-9,limit=600)[0]
c0=3*math.pi*math.sqrt(math.pi/2)/64
c2=2*cross_C(2)/c0
require(abs(c2-9.3352)<.003,'sharp c2')
rec('sharp high-q coefficient c(2)',f'{c2:.8f}')

co=bah.cross_perp_white(2,1,None)/bah.cross_perp_white(0,1,None)
ce=bah.cross_perp_white(2,1,2)/bah.cross_perp_white(0,1,2)
require(abs(co+0.04419)<8e-5 and abs(ce-0.03833)<8e-5,'two-ion correlations')
rec('two-ion common-frequency correlations',f'{co:.6f} / {ce:.6f}')

# B. Corrected curved BEM production suite.
flat=pd.read_csv(ROOT/'screened_bem_flat_validation.csv')
flatmax=float(np.max(np.abs(flat.relative_error)))
require(flatmax<.003,'flat BEM benchmark')
sweep=pd.read_csv(ROOT/'screened_bem_sweep_v2.csv')
require(len(sweep)==16,'sweep size')
r=float(sweep.deltaL.corr(sweep.slope)); rs=float(sweep.deltaL.corr(sweep.slope,method='spearman'))
mae=float(np.mean(np.abs(sweep.slope-sweep.deltaL))); med=float(np.median(np.abs(sweep.rel_error)))
off=sweep[np.abs(sweep.xstar)>.05]; roff=float(off.deltaL.corr(off.slope))
require(r>.998 and rs>.99 and med<.012 and len(off)>=7 and roff>.99,'saddle sweep')
rec('flat BEM max amplitude rel error',f'{100*flatmax:.4f}%')
rec('16-cover Pearson/Spearman',f'{r:.8f} / {rs:.8f}')
rec('16-cover MAE/median rel error',f'{mae:.6f} d / {100*med:.4f}%')
rec('off-axis subset Pearson',f'{roff:.8f} (n={len(off)})')

conv=pd.read_csv(ROOT/'screened_bem_convergence_v2.csv')
flatc=conv[(conv.name=='flat') & np.isclose(conv.fd,.01)]
offc=conv[(conv.name=='offaxis') & np.isclose(conv.fd,.01)]
flatspread=float(flatc.slope.max()-flatc.slope.min()); offspread=float(offc.slope.max()-offc.slope.min())
offd=conv[(conv.name=='offaxis') & np.isclose(conv.target,.11)]
fdspread=float(offd.slope.max()-offd.slope.min())
require(flatspread<.002 and offspread<.004 and fdspread<.001,'BEM convergence')
rec('BEM exponent spreads flat/offaxis/fd',f'{flatspread:.6g} / {offspread:.6g} / {fdspread:.6g}')

row=sweep[sweep.name=='bump_offaxis'].iloc[0]; vertical=1.7114611645805242
require(abs(row.slope-row.deltaL)<.35*abs(row.slope-vertical),'Euclidean discriminator')
rec('off-axis specular/vertical/BEM',f'{row.deltaL:.6f} / {vertical:.6f} / {row.slope:.6f}')

shape=pd.read_csv(ROOT/'billiard_shape_derivative_v2.csv')
shape_rel=float(np.max(np.abs((shape.measured-shape.predicted)/shape.predicted)))
focus=pd.read_csv(ROOT/'billiard_focusing_family_v2.csv'); rf=float(focus.focus.corr(focus.scaled_amp))
multi=pd.read_csv(ROOT/'billiard_multiple_saddle_v2.csv'); rm=float(multi.saddle_sum.corr(multi.scaled_amp))
require(shape_rel<.016 and rf>.999 and rm>.98,'prefactor tests')
rec('shape derivative max rel error',f'{100*shape_rel:.4f}%')
rec('focusing / multiple-saddle correlations',f'{rf:.8f} / {rm:.8f}')

# C. Many-ion covariance and Loewner ordering.
N=10; a=.6
Cy_o=mi.covariance_matrix(N,a,np.inf,0,'normal'); Cy_2=mi.covariance_matrix(N,a,2,0,'normal')
Cx_o=mi.covariance_matrix(N,a,np.inf,0,'x'); Cx_2=mi.covariance_matrix(N,a,2,0,'x')
miny=float(np.linalg.eigvalsh(Cy_2-Cy_o).min()); minx=float(np.linalg.eigvalsh(Cx_o-Cx_2).min())
require(miny>-5e-14 and minx>-5e-14,'representative Loewner PSD')
checks=pd.read_csv(ROOT/'loewner_numerical_checks.csv')
require(checks.min_eig_normal_difference.min()>-5e-14 and checks.min_eig_tangent_difference.min()>-5e-14,'grid Loewner PSD')
r0=mi.participation_rank(Cy_o); r2=mi.participation_rank(Cy_2)
l0,_=mi.eigsorted(Cy_o); l2,_=mi.eigsorted(Cy_2)
lead0=float(l0[0]/l0.sum()); lead2=float(l2[0]/l2.sum())
require(abs(r0-5.61046485)<2e-6 and abs(r2-5.10566497)<2e-6,'rank values')
require(abs(lead0-.2366)<5e-4 and abs(lead2-.2863)<5e-4,'lead shares')
rec('many-ion participation rank open/h2',f'{r0:.6f} / {r2:.6f}')
rec('leading channel share open/h2',f'{lead0:.6f} / {lead2:.6f}')

Cy_xi_o=mi.covariance_matrix(N,a,np.inf,1.0,'normal'); Cy_xi_2=mi.covariance_matrix(N,a,2,1.0,'normal')
rxio=mi.participation_rank(Cy_xi_o); rxi2=mi.participation_rank(Cy_xi_2)
require(abs(rxio-3.48010603)<2e-6 and abs(rxi2-3.02738389)<2e-6,'xi ranks')
rec('xi=d rank open/h2',f'{rxio:.6f} / {rxi2:.6f}')

# Gate-exposure ratios at R=0.6d.
def exposure_ratios(R=.6):
    out={}
    for comp in ['normal','x']:
        Co=mi.covariance_matrix(2,R,np.inf,0,comp); Ch=mi.covariance_matrix(2,R,2,0,comp)
        for lab,b in [('common',[1,1]),('differential',[1,-1])]:
            out[(comp,lab)]=mi.mode_exposure(Ch,b)/mi.mode_exposure(Co,b)
    return out
er=exposure_ratios()
require(abs(er[('normal','common')]-1.237)<.002 and abs(er[('normal','differential')]-1.044)<.002,'normal bus ratios')
require(abs(er[('x','common')]-.878)<.002 and abs(er[('x','differential')]-.967)<.002,'tangent bus ratios')
rec('bus exposure ratios y+/y-/x+/x-', '/'.join(f'{er[k]:.6f}' for k in [('normal','common'),('normal','differential'),('x','common'),('x','differential')]))

# D. Mechanical-mode projection.
nm=pd.read_csv(ROOT/'normal_mode_projection.csv')
means=nm.groupby('geometry').squared_overlap.mean().to_dict()
shares=nm[nm.mechanical_mode==1].set_index('geometry').noise_weight_over_trace.to_dict()
require(abs(means['open']-.872)<.002 and abs(means['h2']-.918)<.002,'mode overlap means')
require(abs(shares['open']-.223)<.002 and abs(shares['h2']-.275)<.002,'common mode shares')
rec('mean mode/environment overlap open/h2',f"{means['open']:.6f} / {means['h2']:.6f}")
rec('common-mode noise share open/h2',f"{shares['open']:.6f} / {shares['h2']:.6f}")

# E. Curved multi-probe slice.
curv=pd.read_csv(ROOT/'curved_multi_ion_covariance.csv')
lead=pd.read_csv(ROOT/'curved_multi_ion_leading_channel.csv')
def mat_for(g):
    z=curv[curv.geometry==g]; n=int(max(z.i.max(),z.j.max())+1); C=np.empty((n,n))
    for rr in z.itertuples(): C[int(rr.i),int(rr.j)]=rr.Cij
    return C
Cf=mat_for('flat'); Cd=mat_for('depressed')
rfp=mi.participation_rank(Cf); rdp=mi.participation_rank(Cd)
x=lead.x_over_d.values; vf=lead.lead_flat.values; vd=lead.lead_depressed.values
cent_f=float(np.sum(x*vf*vf)/np.sum(vf*vf)); cent_d=float(np.sum(x*vd*vd)/np.sum(vd*vd))
require(abs(rfp-2.73162714)<5e-5 and abs(rdp-2.57811715)<5e-5,'curved ranks')
require(abs(cent_f)<2e-4 and abs(cent_d-.421)<.004,'curved centroids')
rec('curved-slice ranks flat/depressed',f'{rfp:.6f} / {rdp:.6f}')
rec('leading-channel centroid flat/depressed',f'{cent_f:.6f} / {cent_d:.6f}')

# F. Primitive MS gate diffusion coefficient and truncation.
gs=pd.read_csv(ROOT/'ms_gate_heating_channel.csv')
floor=float(gs.infidelity.iloc[0]); xx=gs.quanta_per_gate.values[1:4]; yy=gs.infidelity.values[1:4]-floor
m,b=np.polyfit(xx,yy,1)
require(abs(m-.4)<3e-4,'MS gate coefficient')
tr=pd.read_csv(ROOT/'ms_gate_truncation_convergence.csv')
convslopes=tr[tr.nph>=7].slope.values
require(np.max(np.abs(convslopes-.4))<2e-4,'MS truncation')
rec('primitive MS weak-diffusion coefficient',f'{m:.9f}')
rec('nph>=7 slope range',f'{convslopes.min():.9f}--{convslopes.max():.9f}')

# G. Experimental thresholds/counting scale.
chi=1/eta3-1
require(abs(chi-.1092)<1e-4,'cover-noise threshold')
Nnormal=math.ceil(.18/(ry-1)**2); Ntangent=math.ceil(.18/(1-rx)**2)
require(Nnormal==5 and Ntangent==19,'measurement count estimates')
rec('cover-noise threshold chi',f'{chi:.8f}')
rec('ideal 3sigma repeats normal/tangent',f'{Nnormal} / {Ntangent}')

# H. Manuscript consistency and bibliography.
tex=(ROOT/'main_qis.tex').read_text()
require('0.2004' not in tex and '0.400' in tex,'stale gate coefficient')
require('Markov heating term $\\dot{\\bar n}_{\\rm bus}\\mathcal D[a^\\dagger]$' not in tex,'stale one-way gate Lindblad')
require('large-$q$ saddles' not in tex,'ambiguous q notation')
bibs=re.findall(r'\\bibitem\{([^}]+)\}',tex); body=tex.split('\\begin{thebibliography}')[0]
for bib in bibs:
    require(re.search(r'\\cite\{[^}]*\\b'+re.escape(bib)+r'\\b[^}]*\}',body) is not None,f'uncited bibliography item {bib}')
require('2027' not in tex,'post-date reference')
rec('bibliography citation coverage',f'PASS ({len(bibs)} items)')

out='\n'.join(report)+'\nALL FINAL QIS CHECKS PASSED\n'
(ROOT/'verification_output_qis_final.txt').write_text(out)
print(out)
