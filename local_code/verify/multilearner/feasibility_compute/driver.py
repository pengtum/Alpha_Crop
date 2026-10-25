"""按顺序（或 N 路并发的单线程）调用 bench_one.py，逐行追加 JSON（附启动时的系统负载），可断点续跑。
用法：python driver.py <out.jsonl> <threads> <concurrency> <learners> <feats> <ks> <tgts>"""
import itertools, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
out, T, NC = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
L, F, K, C = (sys.argv[i].split(',') for i in range(4, 8))
PY = '/Users/perng_mini/AlphaCrops_local/.venv/bin/python'
HERE = os.path.dirname(os.path.abspath(__file__))
done = set()
if os.path.exists(out):
    for line in open(out):
        r = json.loads(line); done.add((r['learner'], r['feat'], r['k'], r['tgt'], r['threads']))
jobs = [(l, f, int(k), c) for c, k, f, l in itertools.product(C, K, F, L) if (l, f, int(k), c, T) not in done]

def run(j):
    l, f, k, c = j
    la = os.getloadavg()[0]
    t0 = time.time()
    p = subprocess.run([PY, os.path.join(HERE, 'bench_one.py'), l, f, str(k), c, str(T)], capture_output=True, text=True)
    wall = time.time() - t0
    if p.returncode != 0:
        r = dict(learner=l, feat=f, k=k, tgt=c, threads=T, error=p.stderr[-800:])
    else:
        r = json.loads(p.stdout.strip().splitlines()[-1])
    r.update(load1_start=round(la, 2), wall_proc=round(wall, 2), concurrency=NC)
    with open(out, 'a') as fh:
        fh.write(json.dumps(r) + '\n')
    print(l, f, k, c, r.get('t_total', 'ERR'), flush=True)

with ThreadPoolExecutor(NC) as ex:
    list(ex.map(run, jobs))
print('DRIVER DONE', flush=True)
