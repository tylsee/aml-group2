import csv
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import numpy as np
from leaflens.data import load_training_manifest, class_catalogue, find_matches, load_embedding_index

class TrainingGalleryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.img=self.root/'train'
        self.img.mkdir()
        rows=[('a.jpg',0,0,0,'Tomato','Blight','Tomato_Blight'),('b.jpg',0,1,1,'Tomato','Healthy','Tomato_Healthy'),('c.jpg',1,0,2,'Potato','Blight','Potato_Blight')]
        self.rows=rows
        for row in rows:
            Image.new('RGB',(16,16),(190,220,180)).save(self.img/row[0])
        self.manifest=self.root/'train.csv'
        with self.manifest.open('w',newline='') as f:
            csv.writer(f).writerows(rows)
        self.records=load_training_manifest(self.manifest,self.img)
    def tearDown(self):
        self.temp.cleanup()
    def test_catalogue(self):
        self.assertEqual(class_catalogue(self.records),(['Tomato','Potato'],['Blight','Healthy']))
    def test_exact_matches_first(self):
        m=find_matches(self.records,'Tomato','Blight',n=3)
        self.assertEqual([r.record.filename for r in m],['a.jpg','c.jpg','b.jpg'])
        self.assertEqual(m[0].reason,'Exact composition')
        self.assertIsNone(m[0].similarity)
    def test_unseen_pair_fallback(self):
        m=find_matches(self.records,'Potato','Healthy',n=3)
        self.assertEqual([r.record.filename for r in m],['b.jpg','c.jpg'])
        self.assertTrue(all(r.reason!='Exact composition' for r in m))
    def test_train_only_embedding_index(self):
        path=self.root/'index.npz'
        names=np.array([r.filename for r in self.records])
        v=np.eye(3,dtype=np.float32)
        np.savez(path,filenames=names,crop_embeddings=v,disease_embeddings=v)
        index=load_embedding_index(path)
        m=find_matches(self.records,'Tomato','Blight',query_crop_embedding=v[0],query_disease_embedding=v[0],index=index,n=1)
        self.assertAlmostEqual(m[0].similarity,1,places=6)
    def test_escape_attempt_rejected(self):
        with self.manifest.open('a',newline='') as f:
            csv.writer(f).writerow(['../outside.jpg',1,1,3,'Potato','Healthy','Potato_Healthy'])
        with self.assertRaises(ValueError):
            load_training_manifest(self.manifest,self.img)

if __name__=='__main__':unittest.main()
