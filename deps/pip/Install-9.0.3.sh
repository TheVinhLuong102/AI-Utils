#!/bin/bash


pip install pip==9.0.3 --user

if command -v pip2 > /dev/null; then
  pip2 install pip==9.0.3 --user
fi

if command -v pip3 > /dev/null; then
  pip3 install pip==9.0.3 --user
fi
