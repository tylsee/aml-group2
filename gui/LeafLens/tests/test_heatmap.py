import unittest
import numpy as np
from PIL import Image
from leaflens.heatmap import vivid_gradcam

class HeatmapTests(unittest.TestCase):
    def test_heatmap_visible_without_modifying_background(self):
        im=Image.new("RGB",(160,160),(45,125,65))
        y,x=np.mgrid[:160,:160]
        cam=np.exp(-((x-80)**2+(y-64)**2)/(2*16.0**2)).astype("float32")
        out,valid=vivid_gradcam(im,cam)
        pix=np.asarray(out)
        self.assertTrue(valid)
        self.assertGreater(int(pix[64,80,0]),180)
        self.assertLess(int(pix[64,80,1]),125)
        self.assertEqual(tuple(pix[155,155]),(45,125,65))
        self.assertEqual(im.getpixel((80,64)),(45,125,65))
    def test_zero_saliency_is_never_fabricated(self):
        im=Image.new("RGB",(120,90),(35,78,31))
        out,valid=vivid_gradcam(im,np.zeros((12,12),dtype="float32"))
        self.assertFalse(valid)
        self.assertEqual(out.getpixel((30,30)),im.getpixel((30,30)))
    def test_resizes_and_sanitizes(self):
        im=Image.new("RGB",(87,103),(120,160,72))
        cam=np.ones((13,13),dtype="float32")
        cam[0,0]=np.nan
        out,valid=vivid_gradcam(im,cam)
        self.assertTrue(valid)
        self.assertEqual(out.size,im.size)

if __name__=="__main__": unittest.main()
