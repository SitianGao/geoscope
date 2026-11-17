"""Download official RN50 checkpoint with the upstream LFS SHA256 pinned."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlretrieve

RN50_SHA256 = '11657d33a87fc197e19a7654a76a632415ebea42feb8a61025ef389f8990c0f9'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', default='https://huggingface.co')
    parser.add_argument('--output', default='weights/RemoteCLIP-RN50.pt')
    args = parser.parse_args()
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix('.download')
    urlretrieve(args.endpoint.rstrip('/')+'/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-RN50.pt', part)
    with part.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != RN50_SHA256:
            raise ValueError('Downloaded checkpoint checksum mismatch; file was not activated')
    part.replace(path)
    print(str(path), RN50_SHA256)


if __name__ == '__main__': main()
