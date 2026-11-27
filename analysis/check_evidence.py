#!/usr/bin/env python3
"""Check downloaded evidence sizes and SHA-256 hashes before numerical scoring."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--group',action='append',help='Optional group(s), as recorded in the manifest')
    args=parser.parse_args()
    entries=json.loads(args.manifest.read_text())['files']
    if args.group:
        known={r['group'] for r in entries}
        if not set(args.group)<=known:raise ValueError('Unknown evidence group')
        entries=[r for r in entries if r['group'] in args.group]
    for row in entries:
        relative=Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe manifest path')
        path=args.data_root/relative
        if path.stat().st_size!=row['bytes']:raise ValueError(f'Size mismatch: {relative}')
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=row['sha256']:raise ValueError(f'Hash mismatch: {relative}')
    print(json.dumps({'status':'PASS','files_verified':len(entries)}))


if __name__=='__main__':main()
