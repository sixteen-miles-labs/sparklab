"""Correct NVIDIA's ARM wheel platform tag after checking its actual ELF ISA.

The upstream 0.8.1 wheel filename says aarch64, but WHEEL says sbsa, an
unrecognized platform tag. No executable code or dependency versions change.
"""
import base64
import csv
import hashlib
from importlib.metadata import distribution
import io
import platform

assert platform.machine() == 'aarch64'
dist = distribution('nvidia-cusparselt-cu13')
assert dist.version == '0.8.1'
files = list(dist.files)
libraries = [dist.locate_file(f) for f in files if '.so' in f.name]
assert libraries, 'No cuSPARSELT library found'
for path in libraries:
    with path.open('rb') as stream:
        header = stream.read(20)
    assert header[:6] == b'\x7fELF\x02\x01', f'Not ELF64 little-endian: {path}'
    assert int.from_bytes(header[18:20], 'little') == 183, f'Not AArch64: {path}'
wheel = next(f for f in files if str(f).endswith('.dist-info/WHEEL'))
path = dist.locate_file(wheel)
original = path.read_text()
assert 'Tag: py3-none-manylinux2014_sbsa' in original
fixed = original.replace('Tag: py3-none-manylinux2014_sbsa', 'Tag: py3-none-manylinux2014_aarch64').encode()
path.write_bytes(fixed)
record = next(f for f in files if str(f).endswith('.dist-info/RECORD'))
record_path = dist.locate_file(record)
rows = list(csv.reader(io.StringIO(record_path.read_text())))
for row in rows:
    if row[0] == str(wheel):
        row[1] = 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(fixed).digest()).decode().rstrip('=')
        row[2] = str(len(fixed))
with record_path.open('w', newline='') as stream:
    csv.writer(stream).writerows(rows)
print('Verified AArch64 ELF; normalized cuSPARSELT wheel metadata and RECORD')
