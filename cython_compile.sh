#########################################################################
# File Name: cython_compile.sh
# Author: ma6174
# mail: ma6174@163.com
# Created Time: Tue 19 Apr 2016 19:37:44 CST
#########################################################################
#!/bin/bash
[ $# -eq 2 ] && mv ${1}.pyx ${2}.pyx && f=$2
[ $# -eq 1 ] && f=$1
cython ${f}.pyx
gcc -c -fPIC -I /usr/include/python2.7 ${f}.c -w
gcc -shared ${f}.o -o ${f}.so
rm ${f}.[oc]
