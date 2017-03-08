Street Segmentation Toolkit
===========================

This is a collection of tools for pixel-wise street segmentation. Please also consider checking out [KittiSeg](https://github.com/MarvinTeichmann/KittiSeg#kittiseg), a state-of-the art realtime segmentation toolkit.


Installation
------------

Make sure the following packages are installed:

* numpy
* scipy
* lasagne (see http://martin-thoma.com/lasagne-for-python-newbies/)

Install all other packages via

```bash
$ pip install -r requirements.txt --user
```


Now install the package via

```bash
$ make localinstall
```

Make sure that ~/.local/bin is in your PATH.

Now you should be able to execute `sst --help`.

# KittiSeg

Please also check out the [KittiSeg](https://github.com/MarvinTeichmann/KittiSeg#kittiseg) project. It implements similar functionality but which a much higher segmentation accuracy.



