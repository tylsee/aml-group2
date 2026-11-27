from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from PIL import Image
from leaflens.data import TrainRecord
from leaflens.training_gallery import select_references, gallery_html

class TestGallery(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.records = []
        for name, c, d in [('a.jpg','apple','mildew'),('b.jpg','tomato','mildew'),('c.jpg','apple','blight')]:
            path = self.root/name
            Image.new('RGB',(30, 20),'green').save(path)
            self.records.append(TrainRecord(name,0,0,c,d,path))
        self.records.append(TrainRecord('missing.jpg',0,0,'apple','mildew',self.root/'missing.jpg'))
    def tearDown(self):
        self.temp.cleanup()
    def test_exact_only(self):
        count, groups = select_references(self.records,'apple','mildew')
        self.assertEqual(count,2)
        self.assertEqual(len(groups),1)
        self.assertEqual(groups[0][2][0].record.filename,'a.jpg')
        self.assertTrue(groups[0][2][0].image_uri.startswith('data:image/jpeg;base64,'))
    def test_unseen_fallback(self):
        count, groups = select_references(self.records,'tomato','blight')
        self.assertEqual(count,0)
        self.assertEqual([g[2][0].record.filename for g in groups],['c.jpg','b.jpg'])
        self.assertNotIn('Exact crop–disease class', [g[0] for g in groups])
    def test_missing_exact_not_mislabelled(self):
        (self.root/'a.jpg').unlink()
        text = gallery_html(self.records,'apple','mildew',self.root)
        self.assertIn('2 training row(s) match',text)
        self.assertIn('none of their image files could be opened',text)
        self.assertIn('Same disease',text)
    def test_no_files(self):
        for p in self.root.iterdir(): p.unlink()
        text=gallery_html(self.records,'apple','mildew',self.root)
        self.assertIn('No reference photographs could be opened',text)
        self.assertNotIn('data:image/jpeg;base64,',text)
    def test_ignore_unlisted_image(self):
        Image.new('RGB',(30,20),'red').save(self.root/'unlisted.jpg')
        text=gallery_html(self.records,'apple','mildew',self.root)
        self.assertNotIn('unlisted.jpg',text)
if __name__ == '__main__': unittest.main()
