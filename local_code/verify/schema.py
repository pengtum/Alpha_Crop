import pyarrow.parquet as pq, os
D='/Users/perng_mini/AlphaCrops_local/data'
for f in ['dataset_v2.parquet','parcels_eu.parquet']:
    m=pq.ParquetFile(os.path.join(D,f))
    s=m.schema_arrow
    print(f, m.metadata.num_rows, len(s.names))
    print([ (n,str(s.field(n).type)) for n in s.names if not (n.startswith('ae_') or n.startswith('s2_'))])
    print('ae cols', sum(n.startswith('ae_') for n in s.names), 's2 cols', sum(n.startswith('s2_') for n in s.names))
at=set(pq.ParquetFile(os.path.join(D,'s2','s2_AT.parquet')).schema_arrow.names)
print('AT s2 names', len(at))
for sub,pre in [('ae','embeddings_'),('s2','s2_')]:
    for r in ['AT','BE_VLG','BE_WAL','DE_LS','DE_NRW','EE','LT','LV','PT','RO','SE','SI','SK']:
        m=pq.ParquetFile(os.path.join(D,sub,f'{pre}{r}.parquet'))
        names=m.schema_arrow.names
        extra=[n for n in names if n!='parcel_id' and not ((n.startswith('A') and n[1:3].isdigit()) ) ] if sub=='ae' else sorted(set(names)-at)
        missing = sorted(at-set(names)) if sub=='s2' else None
        print(sub, r, m.metadata.num_rows, len(names), 'pid type', m.schema_arrow.field('parcel_id').type, 'extra', extra[:8], len(extra), 'missing', (missing[:5] if missing else missing))
