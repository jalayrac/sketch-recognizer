# Palais de la découverte

This code is used in an exhibition at [Palais de la découverte](http://www.palais-decouverte.fr/en/home/), a french science museum in Paris.

The goal is to introduce people to the concept of Machine Learning.
The module consists of two machines that communicates trough request (using [socket](https://en.wikipedia.org/wiki/Network_socket))
- a machine A where people can draw with a touchpad with a nice interface
- a machine B that receives request from A containing drawings, process the drawing to give class prediction and send response back to A.

## Requirements

Apart from following the installation requirements fo sketch-recognizer, you might need to install additional python libraries (cStringIO, [Pillow](https://pypi.python.org/pypi/Pillow/2.2.1), ...)

Don't forget to edit in `server.py` the location to your caffe installation folder and also select cpu mode or gpu mode (cpu by default).

## Demo

Run in one terminal:

```Shell
python server.py
```

And in another you can send request, an example is given:

```Shell
python client.py
```

If everything works fine, you should expect the following output (in the client terminal):

```Shell
Connection on 4004
[u'zebra', u'tiger', u'horse', u'cow', u'panda']
[0.9689204692840576, 0.031034501269459724, 1.9258541215094738e-05, 1.3325379768502899e-05, 6.308453976089368e-06]
```
