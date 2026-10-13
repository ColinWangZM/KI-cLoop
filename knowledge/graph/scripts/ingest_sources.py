from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
import os
from zipfile import ZipFile
import csv
import io
import json
import re
import time
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(os.environ.get("KICLOOP_GRAPH_WORK", Path(__file__).resolve().parents[1] / "work")).resolve()
ARCHIVE = Path(os.environ.get("KICLOOP_MATERIALS_ARCHIVE", ROOT / "input/materials.zip"))
USPTO = Path(os.environ.get("KICLOOP_USPTO_DIR", ROOT / "input/uspto"))
csv.field_size_limit(100_000_000)
TBPB = re.compile(r'(?:tert[ -]*butyl|t[ -]*butyl|tertiary[ -]*butyl).{0,15}(?:peroxybenzoate|perbenzoate)|(?:peroxybenzoic|perbenzoic).{0,25}butyl|\bTBPB\b', re.I)
BENZ = re.compile(r'\bbenzaldehyde\b', re.I)
PEROX = re.compile(r'hydroperoxide|hypochlorite|peroxybenzoate|perbenzoate', re.I)

def digest(path):
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def process_year(path_string):
    path = Path(path_string)
    year = int(path.stem.replace('output', ''))
    dest = ROOT / 'raw_parquet/uspto' / f'{path.stem}.parquet'
    audit = ROOT / 'audit' / f'{path.stem}.json'
    if dest.exists() and audit.exists():
        return json.loads(audit.read_text())
    dest.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    candidates = {}
    row_count = 0
    malformed = []
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        schema = pa.schema([('source_file', pa.string()), ('source_row', pa.int64()), ('source_year', pa.int32())]
                           + [(name, pa.string()) for name in fields])
        with pq.ParquetWriter(dest, schema, compression='zstd') as writer:
            buffer = []
            for row_number, row in enumerate(reader, 1):
                if None in row:
                    malformed.append({'row': row_number, 'extra_values': row.pop(None)})
                values = {name: row.get(name) for name in fields}
                values.update(source_file=path.name, source_row=row_number, source_year=year)
                buffer.append(values)
                row_count += 1
                product = row.get('product_name') or ''
                operations = row.get('procedure_text') or ''
                flags = []
                if TBPB.search(product):
                    flags.append('product_name_matches_tbpb')
                if BENZ.search(operations) and PEROX.search(operations):
                    flags.append('benzaldehyde_peroxide_or_hypochlorite_in_operations')
                if flags:
                    key = (row.get('ID'), product, operations)
                    if key not in candidates:
                        candidates[key] = {**values, 'match_flags': flags, 'matching_source_rows': [row_number]}
                    else:
                        candidates[key]['matching_source_rows'].append(row_number)
                if len(buffer) >= 25000:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                    buffer.clear()
            if buffer:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
    evidence_file = ROOT / 'procedure_evidence' / f'candidates_{year}.jsonl'
    with evidence_file.open('w') as f:
        for record in candidates.values():
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    report = {'source': str(path), 'source_file': path.name, 'source_bytes': path.stat().st_size,
              'source_sha256': digest(path), 'year': year, 'rows': row_count,
              'parquet': str(dest), 'schema_columns': fields, 'candidate_records': len(candidates),
              'malformed_rows': malformed, 'elapsed_seconds': round(time.time() - start, 2)}
    dump(audit, report)
    return report

def process_materials(dataset):
    start = time.time()
    audit = ROOT / 'audit' / f'materials_{dataset}.json'
    if audit.exists():
        return json.loads(audit.read_text())
    out = ROOT / 'raw_parquet/materials'
    structures = ROOT / 'raw_parquet/structures'
    out.mkdir(parents=True, exist_ok=True)
    structures.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([('dataset', pa.string()), ('record_id', pa.string()),
                        ('property_name', pa.string()), ('value_raw', pa.string()),
                        ('source_member', pa.string()), ('source_row', pa.int64())])
    table_stats = []
    with ZipFile(ARCHIVE) as z:
        prefix = f'crystal_dataset/post_process_dataset/{dataset}/'
        members = sorted(n for n in z.namelist() if n.startswith(prefix) and n.endswith('.csv'))
        for member in members:
            dest = out / f'{dataset}__{Path(member).stem}.parquet'
            total = 0
            hash_ = sha256()
            raw = z.read(member)
            hash_.update(raw)
            with pq.ParquetWriter(dest, schema, compression='zstd') as writer:
                reader = csv.reader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
                header = next(reader)
                assert len(header) == 2, (member, header)
                buffer = []
                for row_number, row in enumerate(reader, 1):
                    if not row:
                        continue
                    assert len(row) == 2, (member, row_number, len(row))
                    buffer.append({'dataset': dataset, 'record_id': row[0], 'property_name': header[1],
                                   'value_raw': row[1], 'source_member': member, 'source_row': row_number})
                    total += 1
                    if len(buffer) >= 25000:
                        writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                        buffer.clear()
                if buffer:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
            table_stats.append({'source_member': member, 'source_sha256': hash_.hexdigest(), 'rows': total,
                                'property_name': header[1], 'parquet': str(dest)})
        print(f'[materials] {dataset}: {len(table_stats)} CSV tables converted; converting CIF structures', flush=True)
        structure_schema = pa.schema([('dataset', pa.string()), ('record_id', pa.string()),
                                      ('source_member', pa.string()), ('source_sha256', pa.string()), ('cif_text', pa.string())])
        cif_members = sorted(n for n in z.namelist() if n.startswith(prefix) and n.endswith('.cif'))
        with pq.ParquetWriter(structures / f'{dataset}.parquet', structure_schema, compression='zstd') as writer:
            buffer = []
            for member in cif_members:
                raw = z.read(member)
                buffer.append({'dataset': dataset, 'record_id': Path(member).stem, 'source_member': member,
                               'source_sha256': sha256(raw).hexdigest(), 'cif_text': raw.decode('utf-8')})
                if len(buffer) >= 10000:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=structure_schema))
                    buffer.clear()
            if buffer:
                writer.write_table(pa.Table.from_pylist(buffer, schema=structure_schema))
    report = {'dataset': dataset, 'tables': table_stats, 'rows': sum(s['rows'] for s in table_stats),
              'cif_records': len(cif_members), 'elapsed_seconds': round(time.time() - start, 2)}
    dump(audit, report)
    return report

def main():
    start = time.time()
    for folder in ["audit", "procedure_evidence", "raw_parquet/uspto", "raw_parquet/materials"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    with ZipFile(ARCHIVE) as z:
        dump(ROOT / 'audit/materials_archive_statistics.json', json.loads(z.read('statistics.json')))
        (ROOT / 'audit/materials_archive_README.txt').write_bytes(z.read('README.txt'))
    print('[start] 41 USPTO yearly CSVs and 4 supplied materials datasets; 12 workers', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=12) as pool:
        jobs = {pool.submit(process_materials, dataset): dataset for dataset in ['oqmd', 'material_project', 'jarvis_data', 'qmof']}
        jobs.update({pool.submit(process_year, str(path)): path.name for path in sorted(USPTO.glob('output*.csv'))})
        for future in as_completed(jobs):
            result = future.result()
            results.append(result)
            print(f'[done {len(results)}/45] {jobs[future]}: rows={result["rows"]:,}; elapsed={time.time()-start:.0f}s', flush=True)
    dump(ROOT / 'audit/ingestion_manifest.json', {'source_archive': str(ARCHIVE), 'archive_sha256': digest(ARCHIVE),
         'archive_bytes': ARCHIVE.stat().st_size, 'jobs': results,
         'elapsed_seconds': round(time.time() - start, 2), 'workers': 12,
         'note': 'Source quantity strings and extracted operation strings are preserved, without unit reinterpretation.'})
    print('[complete] Lossless input conversion and procedure candidate scan finished.', flush=True)

if __name__ == '__main__':
    main()
