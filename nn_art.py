import argparse
import numpy as np
import chainer
from scipy.misc import imread, imresize, imsave
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F
from chainer.functions import caffe
import math
import os

def histogram_match(img, style):
  img_shape = img.shape
  img = img.reshape(-1) #flatten the image matrix
  style = style.reshape(-1)

  u_img, idx, cnt_img = np.unique(img, return_inverse=True, return_counts=True)
  u_style, cnt_style = np.unique(style, return_counts=True)

  img_quantiles = np.cumsum(cnt_img,dtype=np.float32)
  img_quantiles /= img_quantiles[-1]
  style_quantiles = np.cumsum(cnt_style,dtype=np.float32)
  style_quantiles /= style_quantiles[-1]

  val = np.interp(img_quantiles, style_quantiles, u_style)

  return val[idx].reshape(img_shape)

def readImage(path, width):
  image = imread(path)
  w = width
  #h = int(math.floor(image.shape[0]/image.shape[1]*width))
  image = imresize(image, [w,w])
  image = np.transpose(image,(2,0,1))
  image = image.reshape((1,3,w,w))

  return np.ascontiguousarray(image,dtype=np.float32)

def parser_func():
  parser = argparse.ArgumentParser(description='Neural Art')
  parser.add_argument('--content_img', '-c', default='content.png',help='Content image')
  parser.add_argument('--style_img', '-s', default='style.png',help='Style image')
  parser.add_argument('--init_img', '-i', default='noise', help='Set the initial target image')
  parser.add_argument('--width','-w',default=224, type=int, help='Target image width')
  parser.add_argument('--epoch', default=5000, type=int, help='number of epoch')
  parser.add_argument('--step', default=100, type=int, help='step size to store an intermediate image')
  parser.add_argument('--ratio', default=0.001, type=float,help='alpha beta ratio')
  parser.add_argument('--dir', default='result',help='Result Image Directory')
  parser.add_argument('--learningRate', default=4.0, type=float, help='learning rate for optimization')
  return parser
  
def style_layer_forward(model, pic):
  a1,a2,a3,a4,a5, = model(inputs={'data': pic}, outputs=['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1'])

  return a1,a2,a3,a4,a5
  
def content_layer_forward(model, pic):
  return model(inputs={'data': pic}, outputs=['conv4_2'])
  
def style_loss(model, style, target):
  #style loss calculation
  a = style_layer_forward(model, style)
  f = style_layer_forward(model, target)
  M = []
  N = []
  fr = []
  for i in f:
    two =  i.data.shape[2]*i.data.shape[3]
    fr.append(F.reshape(i, (i.data.shape[1], two)))
    M.append(two)
    N.append(i.data.shape[1])

  g = []
  for i in fr:
    g.append(F.matmul(i,i,transa=False,transb=True))
	
  ar = []
  for i in a:
    two =  i.data.shape[2]*i.data.shape[3]
    x = F.reshape(i, (i.data.shape[1], two))
    ar.append(F.matmul(x,x,transa=False,transb=True))
  
  loss = 0
  for i in range(0,5):
    t = F.mean_squared_error(g[i], ar[i])
    loss += t/(M[i]**2 * N[i]**2 * 4)
	
  return loss/5

def loss_fun(model, style, content, target, ratio):
  #content loss calculation
  p = content_layer_forward(model, content)
  f_c = content_layer_forward(model, target)
  dim1 = p[0].data.shape[1]
  MN = p[0].data.shape[2]*p[0].data.shape[3]
  p1 = F.reshape(p[0], (dim1, MN))
  
  dim1 = f_c[0].data.shape[1]
  MN = f_c[0].data.shape[2]*f_c[0].data.shape[3]
  f_c1 = F.reshape(f_c[0], (dim1, MN))
  
  loss_c = F.mean_squared_error(f_c1, p1)/2
  loss_s = style_loss(model, style, target)
  #return loss_s
  return ratio*loss_c + loss_s
  
def saveImage(img, style, filename="canvas.png", width=224):

  img = img.reshape((3,width,width))
  img = np.transpose(img,(1,2,0))
 
  img = histogram_match(img, style)
  imsave(filename,img)  

def main():
  parser = parser_func()
  args = parser.parse_args()
  if os.path.isdir(args.dir)== False:
    os.mkdir(args.dir)
  
  print "Start loading model..."
  model = caffe.CaffeFunction('VGG_ILSVRC_19_layers.caffemodel')
  print "Finish loading model..."
  width = int(args.width)
  if args.init_img == 'noise': #generate a noise image
    target = np.random.uniform(-20, 20, (1, 3, width, width)).astype('float32')
  else:
    target = readImage(args.init_img, width)
	
  style = readImage(args.style_img, width)
  content = readImage(args.content_img, width)
  
  #v_target = Variable(target)
  v_style = Variable(style)
  v_content = Variable(content)
  
  learn = args.learningRate
  target1 = chainer.links.Parameter(target)
  optimizer = optimizers.Adam(alpha=learn)#learning rate
  optimizer.setup(target1)
  for i in range(1, args.epoch+1):
    target1.zerograds()
    tmp = target1.W
    v_target = tmp
    loss = loss_fun(model, v_style, v_content, v_target, args.ratio)
    loss.backward()
    print "loss:", str(loss.data)
    target1.W.grad = tmp.grad
    #print "gradient", target1.W.grad
    optimizer.update()

    if i%args.step==0:
        fname = args.dir + '/img'+str(i)+'.png'
        saveImage(cuda.to_cpu(target1.W.data), style, fname, width)
    print "iteration:", i
  
if __name__ == '__main__':
  main()
