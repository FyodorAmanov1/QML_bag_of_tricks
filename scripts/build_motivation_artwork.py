"""Render motivation cards from recorded original-comparator reproductions."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
summary=json.loads((ROOT/'results/paper_summary.json').read_text(encoding='utf-8'))
alloys=summary['alloys']['fresh_replays']['10']
quantum_accuracy=alloys['quantum']['mean']*100
original_best=max(row['mean'] for name,row in alloys.items() if name!='quantum')*100
communications=summary['communications']
quantum_error=(1-communications['quantum_reservoir']['mean'])*100
original_error=(1-communications['memoryless_logistic']['mean'])*100

fig=plt.figure(figsize=(12.8,7.6),dpi=120,facecolor='#101a32')
ax=fig.add_axes([0,0,1,1]);ax.set_axis_off()
ax.text(.055,.90,'QML applications you can reproduce',fontsize=29,weight='bold',color='white')
ax.text(.055,.84,'Classical inputs  /  Peer-reviewed studies  /  Runnable implementations',fontsize=15,color='#b6c0f3')
for left in (.055,.53):
    ax.add_patch(FancyBboxPatch((left,.27),.415,.50,boxstyle='round,pad=0.015,rounding_size=0.02',
                             facecolor='#1c2b4a',edgecolor='#3f526f',linewidth=1))
for left,label,title,metric,definition,baseline,context in [
    (.075,'MATERIALS SCREENING','Alloy-phase prediction',f'{quantum_accuracy:.2f}%',
     'Quantum SVC accuracy',f'{original_best:.2f}% original classical comparator','10-qubit model  /  10 replay splits'),
    (.55,'CLASSICAL SIGNAL RECOVERY','Wireless channel equalization',f'{quantum_error:.3f}%',
     'Quantum reservoir symbol error',f'{original_error:.3f}% memoryless logistic error','6-qubit model  /  3 fresh datasets at 20 dB')]:
    ax.text(left,.70,label,fontsize=12,weight='bold',color='#83e0ce')
    ax.text(left,.635,title,fontsize=17,color='white')
    ax.text(left,.52,metric,fontsize=38,weight='bold',color='#83e0ce')
    ax.text(left,.465,definition,fontsize=15,color='white')
    ax.text(left,.37,baseline,fontsize=13,color='#e6eafa')
    ax.text(left,.31,context,fontsize=11,color='#b6c0f3')
ax.text(.055,.16,'Original comparator gains in bounded simulations.',fontsize=15,weight='bold',color='white')
ax.text(.055,.105,'Stronger tuned classical models lead in follow-up comparisons; see the complete reports.',fontsize=12,color='#c6cee4')
dest=ROOT/'docs/assets';dest.mkdir(exist_ok=True)
fig.savefig(dest/'qml_motivation.png',facecolor=fig.get_facecolor())
fig.savefig(dest/'qml_motivation.svg',facecolor=fig.get_facecolor())
svg=dest/'qml_motivation.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
plt.close(fig)
print('Rendered application motivation from recorded scores, with comparator and scope labels.')
