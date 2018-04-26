#!/bin/bash


pip uninstall TensorFlow
pip uninstall TensorFlow-GPU

if command -v pip2 > /dev/null; then
  pip2 uninstall TensorFlow
  pip2 uninstall TensorFlow-GPU
fi

if command -v pip3 > /dev/null; then
  pip3 uninstall TensorFlow
  pip3 uninstall TensorFlow-GPU
fi


pip install TensorFlow==1.6.0

if command -v pip2 > /dev/null; then
  pip2 install TensorFlow==1.6.0
fi

if command -v pip3 > /dev/null; then
  pip3 install TensorFlow==1.6.0
fi
