"""Verify and unpack the pinned, unmodified CATLASS header bundle, offline."""
import hashlib
from pathlib import Path
import tarfile

REVISION='769cd40a8716b28650b6bebb08db4834eea4462f'
BUNDLE_SHA256='7f99e4ec0f09f6ef2c5278218faa44785aec7971273e1237778513dc7e137767'


def bundled_headers(root):
    root=Path(root)
    bundle=root/'third_party/catlass_headers.tar.gz'
    if hashlib.sha256(bundle.read_bytes()).hexdigest()!=BUNDLE_SHA256:
        raise RuntimeError('CATLASS bundle checksum mismatch')
    destination=(root/'.deps'/('catlass-'+BUNDLE_SHA256[:12])).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    marker=destination/'archive.sha256'
    if not marker.exists() or marker.read_text()!=BUNDLE_SHA256:
        with tarfile.open(bundle,'r:gz') as archive:
            for member in archive.getmembers():
                path=Path(member.name)
                if (path.is_absolute() or '..' in path.parts or
                    not (path.parts[0]=='include' or member.name=='LICENSE') or
                    not (member.isfile() or member.isdir())):
                    raise RuntimeError('unexpected archive member: '+member.name)
                target=(destination/path).resolve()
                if destination not in target.parents:
                    raise RuntimeError('archive path escapes extraction root')
                if member.isdir():
                    target.mkdir(parents=True,exist_ok=True)
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with archive.extractfile(member) as source:
                        target.write_bytes(source.read())
        marker.write_text(BUNDLE_SHA256)
    return destination
