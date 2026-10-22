"""Synthetic unit fixtures for decontamination rules, never experiment metrics."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from traffic_signs.data import finalize_manifest


def fixture_row(label, split, track, frame, hash_token=None, pixel_token=None):
    """No real image is represented or read by these intentionally synthetic rows."""
    identity = f'synthetic-test-only/{label}/{split}/{track}/{frame}'
    return {'path': identity, 'label':label, 'split':split, 'track':track,
            'sha256':hashlib.sha256((hash_token or identity).encode()).hexdigest(),
            'pixel_sha256':hashlib.sha256((pixel_token or hash_token or identity).encode()).hexdigest()}


class DecontaminationRegression(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'outputs')
        self.addCleanup(self.temporary.cleanup)
        self.config = {'classes':[0,1], 'seed':42, 'output_dir':self.temporary.name}
        self.audit = {'classes':[0,1], 'seed':42, 'corrupt_or_invalid_images':[]}

    def base_rows(self):
        return [fixture_row(label,split,f'base-{label}-{split}',0)
                for label in [0,1] for split in ['train','val','test']]

    def assert_clean_and_covered(self, retained):
        self.assertEqual(set(retained.split), {'train','val','test'})
        for label in [0,1]:
            self.assertEqual(set(retained.loc[retained.label.eq(label),'split']), {'train','val','test'})
        for column in ['sha256','pixel_sha256']:
            self.assertTrue((retained.groupby(column).split.nunique() == 1).all())
        self.assertFalse(set(retained.loc[retained.split.eq('train'),'track']) &
                         set(retained.loc[retained.split.eq('val'),'track']))

    def test_one_matching_frame_quarantines_entire_track_and_preserves_test(self):
        rows = self.base_rows()
        rows += [fixture_row(0,'train','contaminated-track',0,'same-file-as-test'),
                 fixture_row(0,'train','contaminated-track',1),
                 fixture_row(0,'test','additional-official-test',0,'same-file-as-test')]
        original = pd.DataFrame(rows)
        retained = finalize_manifest(original,self.config,self.audit)
        self.assert_clean_and_covered(retained)
        self.assertFalse(retained.track.eq('contaminated-track').any())
        pd.testing.assert_frame_equal(
            original.loc[original.split.eq('test')].reset_index(drop=True),
            retained.loc[retained.split.eq('test'), original.columns].reset_index(drop=True))
        output = Path(self.temporary.name)
        excluded = pd.read_csv(output/'excluded_training_tracks.csv')
        self.assertEqual(len(excluded),2)
        self.assertEqual(set(excluded.track), {'contaminated-track'})
        self.assertEqual(set(excluded.split), {'excluded'})
        self.assertTrue(excluded.exclusion_reason.str.contains('entire_training_track').all())
        audit = json.loads((output/'data_audit.json').read_text(encoding='utf-8'))
        self.assertEqual(audit['excluded_rows'],2)
        self.assertEqual(audit['cross_split_identical_file_groups_before_decontamination'],1)
        self.assertEqual(audit['cross_split_identical_file_groups'],0)
        self.assertEqual(audit['cross_split_identical_rgb_groups'],0)
        self.assertTrue(audit['official_test_rows_unchanged'])
        self.assertEqual(len(pd.read_csv(output/'cross_split_duplicates_initial.csv')),2)
        self.assertEqual(len(pd.read_csv(output/'manifest_with_exclusions.csv')),len(original))

    def test_rgb_match_also_excludes_track_and_train_val_moves_whole_track(self):
        rows = self.base_rows()
        rows += [fixture_row(0,'train','rgb-overlap',0,'file-a','same-rgb-as-test'),
                 fixture_row(0,'train','rgb-overlap',1),
                 fixture_row(0,'test','rgb-test',0,'file-b','same-rgb-as-test'),
                 fixture_row(1,'train','component-train',0,'shared-train-val'),
                 fixture_row(1,'val','component-val',0,'shared-train-val'),
                 fixture_row(1,'val','component-val',1)]
        retained = finalize_manifest(pd.DataFrame(rows),self.config,self.audit)
        self.assert_clean_and_covered(retained)
        self.assertFalse(retained.track.eq('rgb-overlap').any())
        moved = retained[retained.track.eq('component-val')]
        self.assertEqual(len(moved),2)
        self.assertTrue(moved.split.eq('train').all())
        audit = json.loads((Path(self.temporary.name)/'data_audit.json').read_text(encoding='utf-8'))
        self.assertEqual(audit['excluded_tracks'],['rgb-overlap'])
        self.assertEqual(audit['train_val_duplicate_tracks_moved_to_train'],['component-val'])


if __name__ == '__main__':
    unittest.main()
