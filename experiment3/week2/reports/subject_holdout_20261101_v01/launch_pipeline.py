"""Run gates in order, then replace this launcher with the detached experiment."""
from pathlib import Path
import json
import os
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
REPORT=Path(__file__).resolve().parent
SCRIPT=ROOT/'scripts/run_subject_holdout.py'


def stage(name):
    value=json.loads((REPORT/'summary.json').read_text())
    value.update(status='preflight',stage=name,pid=os.getpid())
    (REPORT/'summary.json').write_text(json.dumps(value,indent=2)+'\n')
    print(name,flush=True)


try:
    stage('unit_tests')
    with (REPORT/'runtime_tests.log').open('w') as stream:
        subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_subject_holdout.py','-v'],
                       stdout=stream,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,str(ROOT/'tests/test_eeg_multisubset_v02.py'),'-v'],
                       stdout=stream,stderr=subprocess.STDOUT,check=True)
    for mode in ['bind','gradient','benchmark']:
        stage(dict(bind='source_hash_binding',gradient='real_signal_gradient_check',benchmark='reset_100_step_benchmark')[mode])
        subprocess.run([sys.executable,'-u',str(SCRIPT),mode],check=True)
    os.execv(sys.executable,[sys.executable,'-u',str(SCRIPT),'run'])
except BaseException:
    value=json.loads((REPORT/'summary.json').read_text())
    value.update(status='failed',error=traceback.format_exc())
    (REPORT/'summary.json').write_text(json.dumps(value,indent=2)+'\n')
    raise
