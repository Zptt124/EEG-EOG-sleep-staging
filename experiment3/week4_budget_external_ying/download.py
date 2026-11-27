from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests, re, hashlib, json, time
p=Path('output/budget_external_ying/raw');p.mkdir(parents=True,exist_ok=True)
start=time.monotonic()
u='https://api.osf.io/v2/nodes/gx36m/files/osfstorage/?page%5Bsize%5D=100'
files=[]
while u:
 r=requests.get(u,timeout=30);r.raise_for_status();j=r.json()
 files += [f for f in j['data'] if re.fullmatch(r'SLP\d{3}\.zip|Readme\.txt',f['attributes']['name'],re.I)]
 u=j['links']['next']
assert sum(f['attributes']['name'].startswith('SLP') for f in files)==20
(p/'osf_files.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
print('archive bytes',sum(f['attributes']['size'] for f in files),flush=True)
def download(f):
 a=f['attributes']; dest=p/a['name']; h=a['extra']['hashes']['sha256']
 if dest.exists() and dest.stat().st_size==a['size'] and hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()==h:return
 temp=dest.with_suffix('.partial')
 for i in range(4):
  try:
   with requests.get(f['links']['download'],timeout=(20,45),stream=True) as r:
    r.raise_for_status()
    hash=hashlib.sha256()
    with temp.open('wb') as w:
     for b in r.iter_content(1<<20):
      w.write(b);hash.update(b)
    if temp.stat().st_size!=a['size'] or hash.hexdigest()!=h:raise ValueError('checksum mismatch')
   temp.replace(dest);print(dest.name,round(time.monotonic()-start),flush=True);return
  except Exception as e:
   print(a['name'],'attempt',i+1,type(e).__name__,str(e)[:130],flush=True)
 raise RuntimeError(a['name'])
with ThreadPoolExecutor(8) as e:list(e.map(download,sorted(files,key=lambda f:f['attributes']['name'])))

