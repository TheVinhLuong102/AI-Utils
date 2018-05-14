#!/usr/bin/env bash

# this scripts needs to be called from the Project's root dir


DOCS_DIR=docs
DOCS_BUILD_DIR=$DOCS_DIR/_build


sh $DOCS_DIR/parse-docstr.sh


sphinx-autobuild $DOCS_DIR $DOCS_BUILD_DIR
