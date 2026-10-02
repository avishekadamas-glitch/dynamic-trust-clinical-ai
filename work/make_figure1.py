"""Generate the current conceptual Figure 1; no participant data or document edits."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
W = Path(__file__).resolve().parent
# Original scientific schematic; all links describe a proposed interpretation, not estimated causal effects.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':17})
fig,ax=plt.subplots(figsize=(13.2,12.6));ax.set_xlim(0,13.2);ax.set_ylim(0,12.6);ax.axis('off');fig.subplots_adjust(0,0,1,1)
ink='#172B3A';line='#546875';blue='#EAF1F6';gray='#F3F5F6'
def text(x,y,s,fs=17,**kw):return ax.text(x,y,s,color=ink,fontsize=fs,ha=kw.pop('ha','center'),va=kw.pop('va','center'),**kw)
def box(x,y,w,h,title,body,fc=blue,fs=17):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.06',linewidth=1.2,edgecolor=line,facecolor=fc))
 text(x+w/2,y+h-.28,title,fs,weight='bold')
 text(x+w/2,y+(h-.4)/2,body,fs,linespacing=1.27)
def arrow(x1,y1,x2,y2,dash=False):
 ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=15,linewidth=1.5,color=line,linestyle='--' if dash else '-'))
text(.25,12.27,'A   What may shape beliefs and decisions',21,ha='left',weight='bold')
box(.25,10.48,4.02,1.37,'Person','Prior experience\nTendency to trust • Expertise',fs=17)
box(4.60,10.48,4.02,1.37,'Task and setting','Difficulty • Workload\nConsequences of error',fs=17)
box(8.95,10.48,4.00,1.37,'Experience with this AI','Perceived performance\nFeedback',fs=17)
text(6.6,10.07,'Possible influences from prior literature; their individual effects were not estimated here.',16)
# Bracket relates context to the process without positing one measured latent factor.
ax.plot([.35,.35,12.85,12.85],[9.87,9.72,9.72,9.87],color=line,lw=1)
arrow(6.6,9.72,6.6,9.37,dash=True)
text(.25,9.08,'B   What the proposed framework separates',21,ha='left',weight='bold')
box(.25,6.71,3.83,1.42,'Belief about the system','How sure is the clinician\nthat the AI meets the\nstated performance standard?',fs=17)
box(4.70,6.71,3.80,1.42,'Judgment about this case','How likely is this\nrecommendation to be correct?',fs=17)
box(9.15,6.71,3.80,1.42,'Decision','Use, reject, or seek\nmore information',fs=17)
arrow(4.10,7.37,4.69,7.37,True);arrow(8.53,7.37,9.14,7.37)
text(4.38,8.54,'Needs case information\nand further assumptions',15)
text(11.05,8.54,'Also considers alternatives,\nown confidence, consequences',15)
# Clockwise encounter loop, separated into feedback update and between-encounter change.
arrow(11.05,6.69,11.05,5.73)
box(9.15,4.50,3.80,1.22,'Perceived feedback','What the clinician learns',fc=gray,fs=17)
box(4.70,4.50,3.80,1.22,'Revised belief','After interpreting feedback',fc=gray,fs=17)
box(.25,4.50,3.83,1.22,'Belief at next encounter','After change over time',fc=gray,fs=17)
arrow(9.13,5.13,8.52,5.13);arrow(4.68,5.13,4.10,5.13)
text(8.83,6.02,'Immediate updating',15)
text(4.39,6.02,'Change between encounters',15)
arrow(2.17,5.74,2.17,6.69)
text(1.00,6.20,'Next\nencounter',15)
ax.add_patch(FancyBboxPatch((.25,3.42),12.70,.67,boxstyle='round,pad=0.03',linewidth=1,edgecolor=line,facecolor='white'))
text(6.60,3.755,'A trust rating is an observed response. Its meaning depends on the question, scale, and timing.',16)
text(.25,2.91,'C   What the clinical illustration recorded',21,ha='left',weight='bold')
# Same-screen grouping makes timing uncertainty visible.
for x,w,body in [(.25,2.95,'Case + AI\nrecommendation'),(3.56,2.80,'Accept/reject\nchoice'),(6.72,2.80,'Trust rating\n1–7'),(9.89,3.06,'Correctness\nfeedback')]:
 ax.add_patch(FancyBboxPatch((x,1.48),w,.94,boxstyle='round,pad=0.04,rounding_size=0.06',linewidth=1.2,edgecolor=line,facecolor=gray))
 text(x+w/2,1.95,body,17)
arrow(3.24,1.96,3.50,1.96);arrow(6.41,1.96,6.66,1.96,True);arrow(9.57,1.96,9.83,1.96)
ax.plot([3.56,3.56,9.52,9.52],[1.30,1.16,1.16,1.30],color=line,lw=1)
text(6.54,.83,'Same screen; intended order not confirmed by timestamps.',16)
text(6.60,.32,'The two probabilities in panel B were not directly elicited.',16)
fig.savefig(W/'framework_figure.png',dpi=300,facecolor='white');fig.savefig(W/'framework_figure.svg');plt.close(fig)
