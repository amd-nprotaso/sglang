import datetime
import json
import os
import pathlib
import re
import signal
import subprocess
import time

ROOT = pathlib.Path('/var/home/qwen3_8')
OUT = ROOT / 'logs' / ('tuned_full_sweep_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S'))

def main():
    OUT.mkdir(parents=True)
    server_source = (ROOT / 'run_server.sh').read_text()
    client_source = (ROOT / 'run_bench.sh').read_text()
    configs = {1: {'max-running-requests': 2, 'cuda-graph-max-bs-decode': 2},
               4: {'max-running-requests': 8, 'cuda-graph-max-bs-decode': 8},
               8: {'max-running-requests': 16, 'cuda-graph-max-bs-decode': 16},
               16: {'max-running-requests': 16, 'cuda-graph-max-bs-decode': 16,
                    'page-size': 64, 'chunked-prefill-size': 65536, 'max-mamba-cache-size': 160}}
    for config in configs.values():
        config['mem-fraction-static'] = 0.8
    (OUT / 'configuration.json').write_text(json.dumps({'concurrencies': list(configs), 'overrides': configs, 'duration_seconds': 3600, 'server_source': server_source, 'client_source': client_source}, indent=2))
    print(f'RESULTS={OUT}', flush=True)
    statuses = []
    for conc, config in configs.items():
        directory = OUT / f'results_conc{conc}'
        directory.mkdir()
        server = server_source
        for flag, value in config.items():
            pattern = rf'--{flag}\s+\S+'
            if re.search(pattern, server):
                server = re.sub(pattern, f'--{flag} {value}', server)
            else:
                server = server.replace('--enable-metrics', f'--{flag} {value} ' + chr(92) + '\n--enable-metrics')
        server = server.replace('tee server_conc8.log', f'tee {directory}/server.log')
        client = re.sub(r'source .*?/venvs/agentx/bin/activate', 'source /tmp/qwen_agentx_venv/bin/activate', client_source)
        start = client.index('OUTPUT_DIR=')
        end = client.index('mkdir -p', start)
        client = client[:start] + f'OUTPUT_DIR="{directory}"\n' + client[end:]
        client = client.replace('--concurrency 8', f'--concurrency {conc}')
        client = f'export CONC={conc}\n' + client
        (directory / 'server.sh').write_text(server)
        (directory / 'client.sh').write_text(client)
        subprocess.run(['bash', '-n', str(directory / 'server.sh')], check=True)
        subprocess.run(['bash', '-n', str(directory / 'client.sh')], check=True)
    env = dict(os.environ, HF_HOME='/data/hf_cache', HF_HUB_CACHE='/data/hf_cache/hub')
    for conc in configs:
        directory = OUT / f'results_conc{conc}'
        status = {'concurrency': conc, 'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        print(f'Starting concurrency {conc}', flush=True)
        with (directory / 'launcher.log').open('w') as log:
            process = subprocess.Popen(['bash', str(directory / 'server.sh')], stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            (directory / 'server.pid').write_text(str(process.pid))
            try:
                deadline = time.monotonic() + 1800
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError('Server exited during startup')
                    health = subprocess.run(['curl', '-fsS', '--max-time', '5', 'http://127.0.0.1:18888/health'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if health.returncode == 0:
                        break
                    time.sleep(10)
                else:
                    raise RuntimeError('Server readiness timed out')
                with (directory / 'server_info.json').open('w') as info:
                    subprocess.run(['curl', '-fsS', 'http://127.0.0.1:18888/server_info'], stdout=info)
                print(f'Concurrency {conc}: server ready; starting full 3600-second replay', flush=True)
                with (directory / 'client_launcher.log').open('w') as client_log:
                    result = subprocess.run(['bash', str(directory / 'client.sh')], stdout=client_log, stderr=subprocess.STDOUT, env=env)
                status['client_exit_code'] = result.returncode
            except Exception as exc:
                status['error'] = str(exc)
            finally:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
                status['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                statuses.append(status)
                (OUT / 'status.json').write_text(json.dumps(statuses, indent=2))
                print(json.dumps(status), flush=True)
    summary_source = (ROOT / 'summarize_sweep.py').read_text().replace('(1, 8, 12, 16)', '(1, 4, 8, 16)')
    (OUT / 'summarize.py').write_text(summary_source)
    subprocess.run(['/opt/venv/bin/python', str(OUT / 'summarize.py'), str(OUT)], check=True)
    print('SWEEP FINISHED', flush=True)

if __name__ == '__main__':
    main()
