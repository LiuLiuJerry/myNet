g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /home/jerry/anaconda2/envs/tensorflow/lib/python2.7/site-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/  -O2 -D_GLIBCXX_USE_CXX11_ABI=0