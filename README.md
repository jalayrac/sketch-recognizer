# sketch-recognizer

A simple yet accurate sketch recognizer.
Created by Jean-Baptiste Alayrac at INRIA.

## License

This code is released under the MIT License (refer to the [LICENSE](https://github.com/jalayrac/sketch-recognizer/LICENSE) file for details).

## Contents

  - [Requirements](#requirements)
  - [Run the demo](#run-the-demo)  
  - [Projects using this code](#projects-using-this-code)
  - [Under the hood](#under-the-hood)
  - [Credits](#credits)

## Requirements

This code has been tested under [Ubuntu 16.04](https://wiki.ubuntu.com/XenialXerus/ReleaseNotes?_ga=2.64105455.1187681438.1509407568-1392763737.1509407568),
equipped with a Nvidia GTX1080 (in GPU mode), and CUDA 8.0.
The code may work on previous versions of Ubuntu, but without warranty.
Similarly, earlier or later versions of CUDA should work as well.
Other requirements:

* [python2.7](https://www.python.org/download/releases/2.7/): install with [anaconda](https://www.anaconda.com/download/#download) is always a safe and simple option
* [caffe](http://caffe.berkeleyvision.org/installation.html): follow the installation guide (with python support)

## Run the demo

Once everything is properly installed:

1) Clone this repo and go to the generated folder
  ```Shell
  git clone https://github.com/jalayrac/sketch-recognizer.git
  cd sketch-recognizer
  ```

2) Download the pretrained model:
  ```Shell  
  wget http://www.di.ens.fr/~alayrac/sketch-recognizer/finetune_googledraw_iter_360000.caffemodel -P ./models/  
  ```
3) Setup caffe and CPU/GPU.

Edit the `caffe_root` in `demo.py` to reflect your installation setup (of the form `/path/to/caffe`).
Select if you prefer `gpu` or `cpu` mode (`cpu` by default).

If you have installed caffe with GPU support, don't forget to also add cuda to your `LD_LIBRARY_PATH` (with the path corresponding to your installation):

```Shell 
export LD_LIBRARY_PATH=/usr/cuda-8.0/lib64/:$LD_LIBRARY_PATH
```

4) Run the demo.

  ```Shell
  python samples/*.png
  ```
  
  If everything is setup correctly, you should see the following predictions (after some init messages from caffe):
  
  ```Shell
  sample_1.png: zebra (0.969), tiger (0.03), horse (0.00), cow (0.00), panda (0.00) (served in 0.122 s)
  sample_2.png: sailboat (1.000), canoe (0.00), knife (0.00), chandelier (0.00), submarine (0.00) (served in 0.046 s)
  sample_3.png: banana (0.790), boomerang (0.17), moon (0.01), snake (0.01), trombone (0.00) (served in 0.041 s)
  sample_4.png: wine-bottle (0.992), wineglass (0.01), socks (0.00), lightbulb (0.00), vase (0.00) (served in 0.038 s)
  sample_5.png: eiffel-tower (0.997), skyscraper (0.00), tent (0.00), chandelier (0.00), sword (0.00) (served in 0.041 s)
  ```
  
5) Run on your own images.

To run on your own images, simply creates a free form drawing (with tools such as [this one](https://drawisland.com/?w=400&h=400)), 
save it in png on your computer, and simply type (with the correct path to your drawing):

```Shell
  python /path/to/my/drawing.png
```
  
## Projects using this code

- Palais de la découverte: 
This code has been originally developped for a permanent [exhibition](http://www.palais-decouverte.fr/fr/au-programme/expositions-permanentes/informatique-et-sciences-du-numerique/visite-libre/) in the museum
[Palais de la découverte](http://www.palais-decouverte.fr/en/home/) in Paris.
More specific code used for that project is provided [here](https://github.com/jalayrac/sketch-recognizer/examples/palais/).

- Small web server: see [here](https://github.com/jalayrac/sketch-recognizer/examples/web_server) for an interactive web server that recognize uploaded drawings.

If you happen to use it for your project, please [let me know!](mailto:jean-baptiste.alayrac@inria.fr)

## Under the hood

If you wonder how this model has been obtained, here are some details [TODO].

## Credits

This work wouldn't have been possible without the following great projects:

- [Googlenet](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf)
- All the sketch related research pursued by [James Hays](https://www.cc.gatech.edu/~hays/)'s group.
More precisely, everything started with the following projects:
  - [How Do Humans Sketch Objects?](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/)
  - [The Sketchy Database: Learning to Retrieve Badly Drawn Bunnies](http://sketchy.eye.gatech.edu/): [code here](https://github.com/janesjanes/sketchy).
- Finally, as every deep learning application, one needs lots of data :). Thanks to the [Google Draw](https://quickdraw.withgoogle.com/) project for making
all that great data [available](https://github.com/googlecreativelab/quickdraw-dataset).


I would also like to thank [Francis Bach](http://www.di.ens.fr/~fbach/), [Laurent Viennot](https://who.rocq.inria.fr/Laurent.Viennot/), Vincent Blech (from Palais de la Découverte) and [Fleur De Papier](http://www.fleurdepapier.com/).


