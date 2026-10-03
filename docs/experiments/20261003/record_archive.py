"""Snapshot existing experiment evidence only; never start games or optimization."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
KINDS = ('training', 'pretraining', 'benchmark', 'demonstrations', 'diagnostics')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_csv(name, rows, fields):
    for first in range(0, max(1, len(rows)), 500):
        path = OUT / f'{name}-{first // 500 + 1:02d}.csv'
        with path.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows[first:first + 500])


def partial_collection():
    source = ROOT / 'logs/demonstrations/address-matched-noise-marisa-reimu-20261003'
    manifest = read(source / 'episodes/manifest.json')
    result = read(source / 'result.json')
    identity = read(source / 'identity.json')
    worker = identity['runtime']['game_directory'].replace('\\', '/').split('/')[-2]
    session_path = ROOT.parent / '.dev/workers' / worker / 'session.json'
    session = read(session_path)
    assert not manifest['complete'] and not result['success']
    assert result['error'] == 'KeyboardInterrupt()'
    assert all(session[key] == 0 for key in ('exit_code', 'server_stop_exit', 'server_wait_exit'))
    assert not Path(session['prefix']).exists() and not Path(session['game']).exists()
    pids = (4067413, 4067446, 4067479, 4067669, 4067844)
    assert all(not (Path('/proc') / str(pid)).exists() for pid in pids)
    rows = manifest['episodes']
    for row in rows:
        assert digest(source / 'episodes' / row['path']) == row['sha256']
    groups = {}
    for name, seats in (('overall', (0, 1)), ('player_0', (0,)), ('player_1', (1,))):
        selected = [row for row in rows if row['learner_seat'] in seats]
        outcomes = Counter('win' if row['outcome'] == f"p{row['learner_seat'] + 1}_win" else
            'loss' if row['outcome'] in ('p1_win', 'p2_win') else row['outcome'] for row in selected)
        groups[name] = {'games': len(selected), 'outcomes': dict(outcomes),
            'saved_frames': sum(row['steps'] for row in selected),
            'noise_decisions': sum(row['noise_decisions'] for row in selected),
            'disagreements': sum(row['teacher_behavior_disagreements'] for row in selected),
            'combat_means': {key: sum(row['combat_metrics'][key] for row in selected) / len(selected)
                for key in ('own_hp_loss', 'opponent_hp_loss', 'own_spell_action_entries', 'opponent_spell_action_entries')}}
    summary = {'status': 'interrupted_by_user', 'source': str(source.relative_to(ROOT)),
        'result': result, 'manifest_sha256': digest(source / 'episodes/manifest.json'),
        'result_sha256': digest(source / 'result.json'), 'identity_sha256': digest(source / 'identity.json'),
        'planned_games': 16, 'saved_complete_games': len(rows), 'groups': groups,
        'successful_env_steps_reported': manifest['successful_env_steps'],
        'incomplete_episodes': manifest['incomplete_episodes'],
        'incomplete_observed_frames': sum(row['observed_steps'] for row in manifest['incomplete_episodes']),
        'partial_prefix_storage': 'Only incomplete counters persisted; no complete shard for these episodes.',
        'last_inflight_step': 'attempted_actions exceeds observed_steps by one; not counted as completed evidence',
        'worker': worker, 'session_sha256': digest(session_path), 'session': session,
        'recorded_pids_absent': list(pids), 'cleanup_verified': True,
        'saved_shard_hashes_verified': True, 'teacher_labels_replayed_in_stop_archive': False,
        'full_audit_started': False, 'training_started': False,
        'budget_gate': 'not evaluated on the incomplete 16-game plan; no new training authorized',
        'noise_geometry': 'config/script prepared and committed where applicable; scoring not started'}
    (OUT / 'interrupted-collection.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    assert not (OUT / 'snapshot.json').exists()
    docs = {path: path.read_text() for path in sorted((ROOT / 'docs').glob('*.md'))}
    counts, states, evidence, models = {}, {}, [], []
    errors = []
    for kind in KINDS:
        runs = []
        for directory in sorted((ROOT / 'logs' / kind).iterdir()):
            if not directory.is_dir():
                continue
            relative = directory.relative_to(ROOT).as_posix()
            row = {'path': relative, 'status': 'no_root_result', 'reported_success': '',
                'steps': '', 'best_epoch': '', 'supervised_updates': '', 'error': '',
                'result_sha256': '', 'config_sha256': '', 'identity_sha256': '',
                'documentation': ';'.join(str(p.relative_to(ROOT)) for p, text in docs.items()
                    if relative in text or directory.name in text)}
            for name in ('result', 'config', 'identity'):
                path = directory / (name + ('.yaml' if name == 'config' else '.json'))
                if path.exists():
                    row[name + '_sha256'] = digest(path)
            path = directory / 'result.json'
            if path.exists():
                try:
                    report = read(path)
                    if isinstance(report, dict):
                        if 'success' in report:
                            row['reported_success'] = str(report['success']).lower()
                            row['status'] = 'reported_success' if report['success'] is True else 'reported_failure'
                        else:
                            row['status'] = 'result_without_success_flag'
                        row['error'] = str(report.get('error', ''))
                        payload = report.get('result', report)
                        if isinstance(payload, dict):
                            for key in ('steps', 'best_epoch', 'supervised_updates'):
                                if key in payload and isinstance(payload[key], (int, float, str)):
                                    row[key] = payload[key]
                    else:
                        row['status'] = 'result_non_object'
                except (ValueError, OSError) as error:
                    row.update(status='unreadable_result', error=repr(error))
                    errors.append({'path': str(path.relative_to(ROOT)), 'error': repr(error)})
            if directory.name == 'address-matched-noise-marisa-reimu-20261003' and kind == 'demonstrations':
                row['status'] = 'interrupted_by_user'
            runs.append(row)
            for path in sorted(directory.rglob('*')):
                if not path.is_file():
                    continue
                if path.suffix in ('.json', '.yaml', '.yml', '.csv'):
                    evidence.append({'kind': kind, 'run': relative, 'path': path.relative_to(ROOT).as_posix(),
                        'bytes': path.stat().st_size, 'sha256': digest(path)})
                elif path.suffix == '.zip':
                    # Hash selected model artifacts, list other checkpoints without rereading GBs.
                    hashed = path.name in ('best.zip', 'final.zip', 'initial.zip')
                    models.append({'kind': kind, 'run': relative, 'path': path.relative_to(ROOT).as_posix(),
                        'bytes': path.stat().st_size, 'sha256': digest(path) if hashed else '',
                        'verification': 'rehashed_selected_artifact' if hashed else 'listed_only'})
        counts[kind] = len(runs)
        states[kind] = dict(Counter(row['status'] for row in runs))
        write_csv(kind, runs, ['path', 'status', 'reported_success', 'steps', 'best_epoch',
            'supervised_updates', 'error', 'result_sha256', 'config_sha256', 'identity_sha256', 'documentation'])
    write_csv('evidence', evidence, ['kind', 'run', 'path', 'bytes', 'sha256'])
    write_csv('models', models, ['kind', 'run', 'path', 'bytes', 'sha256', 'verification'])
    document_rows = [{'path': p.relative_to(ROOT).as_posix(), 'title': text.splitlines()[0], 'sha256': digest(p)}
                     for p, text in docs.items()]
    write_csv('documents', document_rows, ['path', 'title', 'sha256'])
    interrupted = partial_collection()
    process_text = subprocess.check_output(['ps', '-eo', 'pid,ppid,user,args'], text=True)
    (OUT / 'processes.txt').write_text('\n'.join(line for line in process_text.splitlines()
        if 'th123-Sudo' in line or 'SokuRL' in line or 'finish-matched-teacher' in line) + '\n')
    script_names = ('finish-matched-teacher-20261003.py', 'audit-matched-teacher-20261003.py',
        'preflight-matched-teacher-20261003.py', 'score-noise-geometry-20261003.py',
        'stop-experiments-20261003.json', 'finish-matched-teacher-20261003.json')
    snapshot = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'purpose': 'User requested stop and documentation; no games, model inference or optimizer runs',
        'run_counts': counts, 'status_counts': states, 'evidence_files': len(evidence), 'model_files': len(models),
        'rehashed_models': sum(row['verification'] == 'rehashed_selected_artifact' for row in models),
        'documents': len(docs), 'parse_errors': errors,
        'excluded_scope': ['logs/hydra duplicate launcher logs', 'logs/validation and deployment (separate runtime acceptance)',
            'binary episode shards/replays indexed by original manifests, not copied into Git',
            'task scratch scripts retained in parent .dev; key current paths fingerprinted below'],
        'status_semantics': 'reported_success is only the raw execution flag, not strength or independent audit approval; no_root_result does not mean running',
        'counts_semantics': 'directories, not independent experiments; diagnostics include nested checks, repeats and shared baselines; never sum as independent games or training budget',
        'interrupted_collection': {'complete_games': interrupted['saved_complete_games'], 'cleanup_verified': True},
        'scratch_evidence': {name: digest(ROOT.parent / '.dev' / name) for name in script_names},
        'record_archive_script_sha256': digest(Path(__file__))}
    (OUT / 'snapshot.json').write_text(json.dumps(snapshot, indent=2))
    print(json.dumps(snapshot, indent=2))


if __name__ == '__main__':
    main()
