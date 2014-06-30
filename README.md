SameImages
==========

Find, delete and rename same images in a directory.


Using OpenCV library.


Using Dirent API for Microsoft Visual Studio (http://softagalleria.net/dirent.php).


By transforming an image into a string, we can create a digital id for each image.


By comparing the ids, we can find similar, even same images.


Then delete the redundant images and rename all rest of the images with its own ids.


You can read about this in http://johnhany.net/2014/06/managing-same-images-with-digital-id/

==========

根据图片像素建立图片的数字ID，据此寻找并删除文件夹中重复的图片，并以数字ID为图片重新命名。


需要OpenCV环境和dirent.h。


可以在这里阅读代码的原理 http://johnhany.net/2014/06/managing-same-images-with-digital-id/
