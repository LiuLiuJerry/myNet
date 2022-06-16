### Usage
try to generate something

### Prerequisites
- Linux (tested under Ubuntu 16.04 )
- Python (tested under 2.7)
- TensorFlow (tested under 1.3.0-GPU )
- numpy, h5py


The code is built on the top of PointNET++. Before run the code, please compile the customized TensorFlow operators of PointNet++ under the folder "pointnet_plusplus/tf_ops".


### Compile Customized TF Operators
The TF operators are included under `tf_ops`, you need to compile them (check `tf_xxx_compile.sh` under each ops subfolder) first. Update nvcc and python path if necessary. The code is tested under TF1.2.0. If you are using earlier version it's possible that you need to remove the `-D_GLIBCXX_USE_CXX11_ABI=0` flag in g++ command in order to compile correctly.


To compile the operators in TF version >=1.4, you need to modify the compile scripts slightly.


First, find Tensorflow include and library paths.
```
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
```

Then, add flags of -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework to the g++ commands.


Note that installing  tensorflow in conda vitual environments could cause some .so file not finding problem
