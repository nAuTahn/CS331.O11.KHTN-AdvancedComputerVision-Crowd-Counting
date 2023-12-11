import MCNN
import GLoss
import VGG19
import Res50
from MCNN import testMCNN
from GLoss import testGLoss
from VGG19 import testVGG19
from Res50 import testRes50

def test(img):
    return testMCNN.test(img), testGLoss.test(img), testVGG19.test(img), testRes50.test(img)
        
