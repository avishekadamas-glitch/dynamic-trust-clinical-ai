"""Generate the manuscript mathematical examples (Figure 2); no participant data."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','mathtext.fontset':'dejavusans'})
blue='#285B84'; teal='#247A73'; ink='#172B3A'; gray='#61707B'; orange='#B26722'
fig,axs=plt.subplots(1,3,figsize=(8.0,3.7),gridspec_kw={'width_ratios':[1,1.25,1]})
ax=axs[0]
x=np.arange(2);width=.32
ax.bar(x-width/2,[.5,.5],width,label='Capability belief T',color=blue)
ax.bar(x+width/2,[.8,.5],width,label='Case confidence q',color=teal)
ax.set_xticks(x,['Distribution A','Distribution B'],fontsize=8);ax.set_ylim(0,1);ax.set_ylabel('Subjective probability');ax.set_title('A  Capability and prediction',loc='left',fontsize=10.5,fontweight='bold')
for xx,yy in zip(x-width/2,[.5,.5]):ax.text(xx,yy+.025,f'{yy:.2f}',ha='center',fontsize=8)
for xx,yy in zip(x+width/2,[.8,.5]):ax.text(xx,yy+.025,f'{yy:.2f}',ha='center',fontsize=8)
ax.legend(frameon=False,loc='upper left',fontsize=8.5,bbox_to_anchor=(0,-.19))
a=np.log(3);sig=lambda z:1/(1+np.exp(-np.array(z)))
ax=axs[1]
ax.plot([0,1,2],sig([0,a,-a/2]),'-o',color=orange,label='+ then −, φ = 0.5',markersize=4)
ax.plot([0,1,2],sig([0,-a,a/2]),'-s',color=teal,label='− then +, φ = 0.5',markersize=4)
ax.plot([0,1,2],sig([0,a,0]),'--',color=orange,alpha=.6,label='Coherent, φ = 1')
ax.plot([0,1,2],sig([0,-a,0]),'--',color=teal,alpha=.6)
ax.scatter([2],[.5],facecolor='white',edgecolor=gray,zorder=5,s=22)
ax.set_xticks([0,1,2],['Initial','After 1','After 2'],fontsize=8);ax.set_ylim(0,1);ax.set_title('B  Evidence order',loc='left',fontsize=10.5,fontweight='bold')
ax.legend(frameon=False,loc='upper left',fontsize=8.3,bbox_to_anchor=(0,-.19))
ax=axs[2];ax.axhline(.65,ls='--',color=gray,lw=1)
ax.scatter([0,1],[.8,.5],c=[teal,orange],s=70,zorder=4)
ax.text(0,.85,'Rely',ha='center',fontsize=9,color=teal);ax.text(1,.41,'Override',ha='center',fontsize=9,color=orange)
ax.text(.5,.68,'Threshold = 0.65',ha='center',fontsize=8,color=gray)
ax.set_xticks([0,1],['Distribution A','Distribution B'],fontsize=8);ax.set_xlim(-.4,1.4);ax.set_ylim(0,1);ax.set_title('C  Optimal actions',loc='left',fontsize=10.5,fontweight='bold')
for ax in axs:ax.set_yticks([0,.25,.5,.75,1]);ax.tick_params(axis='y',labelsize=8);ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
fig.subplots_adjust(left=.065,right=.99,top=.87,bottom=.34,wspace=.32)
out = Path(__file__).resolve().parent
fig.savefig(out/'figure2.png',dpi=300,facecolor='white');fig.savefig(out/'figure2.svg');plt.close(fig)
assert np.isclose(sig(-a/2),1/(1+np.sqrt(3)))
print('Verified calculations:',sig([-a/2,a/2]).tolist(), 'capability',[.5,.5], 'predictive',[.8,.5])
