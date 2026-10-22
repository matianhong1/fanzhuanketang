"""Read official GTSRB PPM annotations and create a deterministic track split."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
from io import BytesIO
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from .labels import CLASS_NAMES

SOURCE = 'https://github.com/houbensebastian/GermanTrafficSignBenchmarks'
ARCHIVE_BASE = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/'

def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # Reject pre-existing filesystem links once. Lexical ZIP-name validation below
    # avoids resolving tens of thousands of absent files on Windows.
    for root, dirs, files in os.walk(destination, followlinks=False):
        for name in dirs + files:
            candidate = os.path.join(root, name)
            if os.path.islink(candidate) or getattr(os.path,'isjunction',lambda _:False)(candidate):
                raise ValueError(f'Existing filesystem link in extraction directory: {candidate}')
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            member = PurePosixPath(item.filename)
            if member.is_absolute() or '..' in member.parts or '\\' in item.filename or ':' in item.filename:
                raise ValueError(f'Unsafe archive path: {item.filename}')
            if ((item.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError('Archive symbolic links are not allowed')
        z.extractall(destination)

def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def prepare_data(config):
    data = Path(config.get('data_dir', 'data')).resolve()
    extracted = data / 'extracted'
    required = ['GTSRB_Final_Training_Images.zip', 'GTSRB_Final_Test_Images.zip', 'GTSRB_Final_Test_GT.zip']
    for filename in required:
        marker = extracted / (filename + '.extracted')
        if not marker.exists():
            archive = data / 'raw' / filename
            if not archive.exists():
                raise FileNotFoundError(f'Missing {archive}. Run python scripts/download_data.py')
            print(f'Extracting {filename} ...', flush=True)
            safe_extract(archive, extracted)
            marker.write_text(_sha(archive), encoding='utf-8')
    classes = sorted(int(x) for x in config['classes'])
    if not classes or not set(classes) <= set(CLASS_NAMES):
        raise ValueError('classes must contain valid original IDs 0..42')
    train_csvs = sorted(extracted.glob('**/GT-?????.csv'))
    test_csvs = list(extracted.glob('**/GT-final_test.csv'))
    if len(train_csvs) != 43 or len(test_csvs) != 1:
        raise ValueError(f'Unexpected annotation structure: {len(train_csvs)} train CSVs, {len(test_csvs)} test CSVs')
    rows = []
    for csv in train_csvs:
        for r in pd.read_csv(csv, sep=';').to_dict('records'):
            if int(r['ClassId']) not in classes:
                continue
            rows.append(_row(r, csv.parent / r['Filename'], 'train'))
    test_dir = extracted / 'GTSRB' / 'Final_Test' / 'Images'
    for r in pd.read_csv(test_csvs[0], sep=';').to_dict('records'):
        if int(r['ClassId']) in classes:
            rows.append(_row(r, test_dir / r['Filename'], 'test'))
    manifest = pd.DataFrame(rows)
    rng = np.random.default_rng(config.get('seed',42))
    fraction = float(config.get('validation_fraction',.2))
    if not 0 < fraction < 1:
        raise ValueError('validation_fraction must be between 0 and 1')
    val_tracks = []
    for label in classes:
        tracks = sorted(manifest.loc[(manifest.label == label) & (manifest.split == 'train'),'track'].unique())
        if len(tracks) < 2:
            raise ValueError(f'Class {label} needs at least two tracks')
        n = min(len(tracks)-1, max(1, int(round(len(tracks)*fraction))))
        val_tracks.extend(rng.choice(tracks,n,replace=False).tolist())
    manifest.loc[manifest.track.isin(val_tracks), 'split'] = 'val'
    errors, hashes, pixel_hashes = [], [], []
    print(f'Auditing {len(manifest)} decoded images and exact hashes ...', flush=True)
    for number, r in enumerate(manifest.itertuples(), 1):
        try:
            content = Path(r.path).read_bytes()
            with Image.open(BytesIO(content)) as im:
                im.load()
                if im.size != (r.width,r.height):
                    raise ValueError('image dimensions disagree with annotations')
                if not (0 <= r.x1 <= r.x2 < r.width and 0 <= r.y1 <= r.y2 < r.height):
                    raise ValueError('invalid ROI')
                pixel_hashes.append(hashlib.sha256(str(im.size).encode()+im.convert('RGB').tobytes()).hexdigest())
            hashes.append(hashlib.sha256(content).hexdigest())
        except Exception as exc:
            errors.append({'path':r.path,'error':str(exc)})
            hashes.append('invalid'); pixel_hashes.append('invalid')
        if number % 10000 == 0:
            print(f'Data audit: {number}/{len(manifest)} images', flush=True)
    manifest['sha256'] = hashes
    manifest['pixel_sha256'] = pixel_hashes
    output = Path(config['output_dir']); output.mkdir(parents=True,exist_ok=True)
    train_tracks = set(manifest.loc[manifest.split=='train','track'])
    val_tracks_set = set(manifest.loc[manifest.split=='val','track'])
    cross = manifest.groupby('sha256').split.nunique()
    pixel_cross = manifest.groupby('pixel_sha256').split.nunique()
    audit = {'source':SOURCE,'official_download_base':ARCHIVE_BASE,
             'archives':{f:_sha(data/'raw'/f) for f in required},
             'seed':int(config.get('seed',42)), 'classes':classes,
             'track_rule':'training: original class ID + filename prefix before underscore; complete physical-sign tracks assigned within each class',
             'split_rule':'per-class seeded choice of round(track_count * validation_fraction), at least one validation and one training track; official test unchanged',
             'official_test_track_metadata_available':False,
             'train_val_track_overlap':len(train_tracks & val_tracks_set),
             'cross_split_identical_file_groups':int((cross>1).sum()),
             'cross_split_identical_rgb_groups':int((pixel_cross>1).sum()),
             'corrupt_or_invalid_images':errors, 'counts':manifest.groupby('split').size().to_dict(),
             'original_dimensions':{'min_width':int(manifest.width.min()),'max_width':int(manifest.width.max()),'min_height':int(manifest.height.min()),'max_height':int(manifest.height.max())},
             'preprocessing':{'image_size':config.get('image_size',32),'use_roi':config.get('use_roi',False),'augmentation':False},
             'limitation':'Test filenames do not expose physical-sign track IDs; no explicit cross-set physical identity verification beyond official separation and exact image hashes.'}
    return finalize_manifest(manifest, config, audit)

def finalize_manifest(manifest, config, initial_checks):
    """Finalize an already decoded/hashed manifest without re-reading images.

    This function is intentionally explicit: callers must supply the completed
    audit from prepare_data, never a manifest with unknown integrity. Structural
    decontamination uses identity hashes only, not predictions or test labels.
    """
    manifest = manifest.copy()
    audit = dict(initial_checks)
    classes = sorted(int(c) for c in config['classes'])
    required = {'path','label','split','track','sha256','pixel_sha256'}
    if not required <= set(manifest.columns):
        raise ValueError('Finalization requires the fully hashed manifest')
    if audit.get('corrupt_or_invalid_images'):
        raise ValueError('Cannot finalize a manifest containing invalid images')
    if audit.get('seed') != config.get('seed',42) or audit.get('classes') != classes:
        raise ValueError('Audited manifest seed/classes disagree with configuration')
    if not manifest.split.isin(['train','val','test']).all():
        raise ValueError('Finalize requires original pre-quarantine train/val/test rows')
    for column in ['sha256','pixel_sha256']:
        if not manifest[column].astype(str).str.fullmatch('[0-9a-f]{64}').all():
            raise ValueError('Missing or malformed audited image hashes')
    output = Path(config['output_dir']); output.mkdir(parents=True,exist_ok=True)
    audit['counts_before_decontamination'] = manifest.groupby('split').size().to_dict()
    audit['cross_split_identical_file_groups_before_decontamination'] = int((manifest.groupby('sha256').split.nunique()>1).sum())
    audit['cross_split_identical_rgb_groups_before_decontamination'] = int((manifest.groupby('pixel_sha256').split.nunique()>1).sum())
    duplicate_mask = pd.Series(False,index=manifest.index)
    for column in ['sha256','pixel_sha256']:
        overlap = manifest.groupby(column).split.nunique()
        duplicate_mask |= manifest[column].isin(overlap[overlap>1].index)
    manifest.loc[duplicate_mask].to_csv(output/'cross_split_duplicates_initial.csv',index=False,encoding='utf-8-sig')
    manifest['original_split'] = manifest.split
    manifest['exclusion_reason'] = ''
    is_test = manifest.split.eq('test')
    matched = pd.Series(False,index=manifest.index)
    for column in ['sha256','pixel_sha256']:
        matched |= ~is_test & manifest[column].isin(set(manifest.loc[is_test,column]))
    excluded_tracks = sorted(manifest.loc[matched,'track'].unique())
    excluded_mask = ~is_test & manifest.track.isin(excluded_tracks)
    manifest.loc[excluded_mask,'split'] = 'excluded'
    manifest.loc[excluded_mask,'exclusion_reason'] = 'entire_training_track_contains_exact_image_matching_official_test'
    # If two training-pool tracks share exact images across train/val, move the
    # whole connected validation component to train until no overlap remains.
    moved_tracks = set()
    while True:
        active = manifest[manifest.split.isin(['train','val'])]
        to_move = set()
        for column in ['sha256','pixel_sha256']:
            crossing = active.groupby(column).split.nunique()
            to_move.update(active.loc[active.split.eq('val') & active[column].isin(crossing[crossing>1].index),'track'])
        if not to_move: break
        manifest.loc[manifest.split.eq('val') & manifest.track.isin(to_move),'split'] = 'train'
        moved_tracks.update(to_move)
    retained = manifest[manifest.split.ne('excluded')].copy()
    for label in classes:
        if set(retained.loc[retained.label.eq(label),'split']) != {'train','val','test'}:
            raise ValueError(f'Decontamination left class {label} without all three retained splits')
    audit['decontamination_policy'] = 'Keep official test unchanged; quarantine entire training-pool tracks with a file or decoded-RGB hash matching any official test image. Do not use test predictions or labels to select exclusions. Move whole validation tracks into train if exact-image components connect train and val.'
    audit['excluded_tracks'] = excluded_tracks
    audit['excluded_rows'] = int(excluded_mask.sum())
    audit['train_val_duplicate_tracks_moved_to_train'] = sorted(moved_tracks)
    audit['official_test_rows_unchanged'] = bool(manifest.loc[is_test,'split'].eq('test').all())
    audit['counts'] = retained.groupby('split').size().to_dict()
    audit['counts_including_excluded'] = manifest.groupby('split').size().to_dict()
    audit['train_val_track_overlap'] = len(set(retained.loc[retained.split.eq('train'),'track']) & set(retained.loc[retained.split.eq('val'),'track']))
    audit['cross_split_identical_file_groups'] = int((retained.groupby('sha256').split.nunique()>1).sum())
    audit['cross_split_identical_rgb_groups'] = int((retained.groupby('pixel_sha256').split.nunique()>1).sum())
    audit['limitation'] = 'Exact hash matches caused whole-track quarantine. Non-identical near-duplicate physical instances cannot be ruled out because official test filenames do not expose physical-sign track IDs.'
    manifest.to_csv(output/'manifest_with_exclusions.csv',index=False,encoding='utf-8-sig')
    manifest.loc[excluded_mask].to_csv(output/'excluded_training_tracks.csv',index=False,encoding='utf-8-sig')
    retained.to_csv(output/'manifest.csv',index=False,encoding='utf-8-sig')
    (output/'data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    counts = retained.groupby(['label','split']).size().unstack(fill_value=0).reindex(classes)
    counts['name_zh'] = [CLASS_NAMES[c] for c in counts.index]
    counts.to_csv(output/'class_counts.csv',encoding='utf-8-sig')
    (output/'label_mapping.json').write_text(json.dumps({c:CLASS_NAMES[c] for c in classes},ensure_ascii=False,indent=2),encoding='utf-8')
    if audit['train_val_track_overlap'] or audit['cross_split_identical_file_groups'] or audit['cross_split_identical_rgb_groups']:
        raise ValueError(f'Data integrity/leakage checks failed; see {output}/data_audit.json')
    return retained

def _row(r,path,split):
    label = int(r['ClassId'])
    return {'path':os.path.abspath(path),'label':label,'split':split,
            'track':f"{label:05d}_{Path(r['Filename']).stem.split('_')[0]}" if split=='train' else 'test_'+r['Filename'],
            'x1':int(r['Roi.X1'] if 'Roi.X1' in r else r['ROI.x1']),
            'y1':int(r['Roi.Y1'] if 'Roi.Y1' in r else r['ROI.y1']),
            'x2':int(r['Roi.X2'] if 'Roi.X2' in r else r['ROI.x2']),
            'y2':int(r['Roi.Y2'] if 'Roi.Y2' in r else r['ROI.y2']),
            'width':int(r['Width']),'height':int(r['Height'])}
