#!/usr/bin/env bash


DOCS_BUILD_DIR=docs/_build


cd $DOCS_BUILD_DIR

tar cz * | ssh ubuntu@doc.arimo.com "cd /var/www/html/BAI-dev; tar zxvf -;"
