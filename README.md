# videograph: a package to create/manipulate graphs on videos

This package provides standard functions to
create and manipulate edge-weighted graphs 
of videos: create a graph, segment it, get 
its adjacency matrix, ...

## Install 

1/ Torch7 is required:

Dependencies, on Linux (Ubuntu > 9.04):

``` sh
$ apt-get install gcc g++ git libreadline5-dev cmake wget libqt4-core libqt4-gui libqt4-dev
```

Dependencies, on Mac OS (Leopard, or more), using [Homebrew](http://mxcl.github.com/homebrew/):

``` sh
$ brew install git readline cmake wget qt
```

Then on both platforms:

``` sh
$ git clone https://github.com/andresy/torch
$ cd torch
$ mkdir build; cd build
$ cmake ..
$ make
$ [sudo] make install
```

2/ Once Torch7 is available, install this package:

``` sh
$ [sudo] torch-pkg install videograph
```

## Use the library

First run torch, and load videograph:

``` sh
$ torch
``` 

``` lua
> require 'videograph'
```

...