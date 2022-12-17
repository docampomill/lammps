#!/usr/bin/env python

from __future__ import print_function
import sys, os, platform, subprocess, shutil
from argparse import ArgumentParser

sys.path.append('..')
from install_helpers import geturl, fullpath

parser = ArgumentParser(prog='Install.py',
                        description="LAMMPS library build wrapper script")

# help message

HELP = """
Syntax from src dir: make lib-tensorflow args="-b"
                 or: make lib-tensorflow args='-p /usr/local'

Syntax from lib dir: python Install.py -b
                 or: python Install.py -p /usr/local 

Example:

make lib-tensorflow args="-b"   # download tensorflow c library in lib/tensorflow
make lib-tensorflow args="-p /usr/local" # use existing tensorflow c library in /usr/local

TensorFlow C library can be downloaded from "https://www.tensorflow.org/install/lang_c"                 
"""

pgroup = parser.add_mutually_exclusive_group()
pgroup.add_argument("-b", "--build", action="store_true",
                    help="download TensorFlow C library")
pgroup.add_argument("-p", "--path",
                    help="specify folder of existing TensorFlow C library")


args = parser.parse_args()

# cmd = """
# FILENAME=libtensorflow-cpu-linux-x86_64-2.11.0.tar.gz && \

# wget -q --no-check-certificate https://storage.googleapis.com/tensorflow/libtensorflow/${FILENAME} || { echo "please install wget or download TensorFlow C library from https://www.tensorflow.org/install/lang_c and put it into lammps/lib/tensorflow directory"; exit; } && \

# tar -xzf ${FILENAME} && \

# rm $FILENAME
# """

# print help message and exit, if neither build nor path options are given
if not args.build and not args.path:
  parser.print_help()
  sys.exit(HELP)

buildflag = args.build
pathflag = args.path is not None
tfpath = args.path

homepath = fullpath('.')
homedir = "%s" % (homepath)


if pathflag:
    if not os.path.isdir(tfpath):
      sys.exit("Tensorflow C library path %s does not exist" % tfpath)
    homedir = fullpath(tfpath)
    if not os.path.isfile(os.path.join(homedir, 'lib', 'libtensorflow.so.2')):
      sys.exit("No Tensorflow C library found at %s" % tfpath)

  
if buildflag:
  url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.11.0.tar.gz"
  filename = "tflib.tar.gz" 
  print("Downloading TensorFlow C library ...")
  geturl(url, filename)

  print("Unpacking TensorFlow C library tarball ...")
  cmd = 'tar -xzvf %s' % filename
  subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
  os.remove(filename)


# create 2 links in lib/tensorflow to Tensorflow C library dir

print("Creating links to Tensorflow C library include and lib files")
if os.path.isfile("includelink") or os.path.islink("includelink"):
  os.remove("includelink")
if os.path.isfile("liblink") or os.path.islink("liblink"):
  os.remove("liblink")
os.symlink(os.path.join(homedir, 'include'), 'includelink')
os.symlink(os.path.join(homedir, 'lib'), 'liblink')


print('\n***************************************    Action required    **************************************************') 
print('Please configure the linker environmental variables:')

libpath = os.path.join(homedir,'lib')
print('export LIBRARY_PATH=$LIBRARY_PATH:'+libpath)
print('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'+libpath)
print('****************************************************************************************************************')


