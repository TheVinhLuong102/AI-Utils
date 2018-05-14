#!/usr/bin/env bash


DOCS_BUILD_DIR=docs/_build
PUBLICATION_DIR=ubuntu@doc.arimo.com:/var/www/html/BAI-dev


rsync -avr $DOCS_BUILD_DIR/ $PUBLICATION_DIR/ --delete
