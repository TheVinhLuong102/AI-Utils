#!/usr/bin/env bash

# this scripts needs to be called from the Project's root dir


MODULE=arimo
DOCS_DIR=docs


# generate .rst files from module code & docstrings
# any pathnames given at the end are paths to be excluded ignored during generation.
# ref: http://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html
sphinx-apidoc --force --maxdepth 4 --module-first --output-dir $DOCS_DIR --separate $MODULE

# get rid of undocumented members
# grep -C2 ":undoc-members:" $DOCS_DIR/$MODULE*.rst
sed -e /:undoc-members:/d -i .orig $DOCS_DIR/$MODULE*.rst
rm $DOCS_DIR/*.orig
