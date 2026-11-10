"""Execute and save notebooks with a workspace-local, explicit Python kernel."""
import json,os,sys,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
cache=ROOT/'.cache/jupyter';kernel=cache/'kernels/qml-reference';kernel.mkdir(parents=True,exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='QML reference',language='python')))
os.environ['JUPYTER_PATH']=str(cache);os.environ['JUPYTER_RUNTIME_DIR']=str(cache/'runtime')
os.environ['IPYTHONDIR']=str(cache/'ipython')
os.environ['MPLCONFIGDIR']=str(cache/'matplotlib')
if sys.platform=='win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
Path(os.environ['JUPYTER_RUNTIME_DIR']).mkdir(exist_ok=True)
rows=[]
for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
    nb=nbformat.read(path,as_version=4);started=time.perf_counter()
    NotebookClient(nb,timeout=300,kernel_name='qml-reference',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,path)
    rows.append(dict(notebook=path.name,seconds=time.perf_counter()-started,executed_code_cells=sum(c.cell_type=='code' for c in nb.cells),errors=[]))
    print(f'Executed {path.name}: {rows[-1]["seconds"]:.1f}s',flush=True)
(ROOT/'results/notebook_execution.json').write_text(json.dumps(dict(python=sys.version,executable_scope='workspace-local validation kernel',runs=rows),indent=2)+'\n')
