"""启动独立后台流程；防止空闲休眠，日志写入新实验目录。"""
from pathlib import Path
import os,sys,subprocess,json,time,fcntl
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run as R
if __name__=='__main__':
    frozen=R.prepare()
    with (R.ROOT/'pipeline.lock').open('a') as lk:
        try:fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Pipeline already running; not launching another copy.')
        fcntl.flock(lk,fcntl.LOCK_UN)
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR='/private/tmp/alpha_mpl',XDG_CACHE_HOME='/private/tmp/alpha_cache')
    with (R.ROOT/'pipeline.log').open('a',buffering=1) as log:
        child=subprocess.Popen(['/usr/bin/caffeinate','-i',sys.executable,'-B',str(HERE/'run.py'),'pipeline'],cwd=R.PROJECT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    R.atomic_json(R.ROOT/'launch.json',dict(supervisor_pid=child.pid,launched=time.time(),code_hash=frozen['code_hash']))
    print('Started background pipeline, supervisor PID',child.pid)
