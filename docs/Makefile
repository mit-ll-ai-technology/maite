# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Tox specific variables
# tox_html target is expected to be invoked from tox in the docs env where changedir=docs
TOXENVDIR        ?= .
TOXENVTMPDIR     ?= ./tmp

IPYNB_DIR := ../examples
RST_DIR    := build/generated

IPYNB_PATTERN := $(IPYNB_DIR)/*/*.ipynb
RST_PATTERN = $(RST_DIR)/*/*.rst

IPYNB_FILES := $(wildcard $(IPYNB_PATTERN))
RST_FILES := $(foreach file,$(IPYNB_FILES),$(RST_DIR)/$(subst $(IPYNB_DIR)/,,$(file:.ipynb=.rst)))

# Put it first so that "make" without argument is like "make help".
# ('make' without arguments defaults to first target)
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean notebook_rst tox_html

clean:
	@rm -r "$(BUILDDIR)"

# Make cheat sheet
# $< First prerequisite
# $@ Target being generated

# Make sure the .ipynb files do not trigger the catch-all target
$(IPYNB_FILES):
	echo ipynb target

$(RST_DIR)/%.rst: $(IPYNB_DIR)/%.ipynb
	jupyter nbconvert --log-level=WARN --execute --to rst $< --output-dir $(dir $@)

notebook_rst: $(RST_FILES)

# EA: I'm not up on sphinx, the command below reflects the origin tox.ini command. Why -M html vs -b html? 
# We could use the original target setting SOURCEDIR, BUILDDIR, and SPHINXOPTS='-d "$(TOXENVTMPDIR)"/doctrees'
tox_html: $(RST_FILES)
	sphinx-build -b html -d "$(TOXENVTMPDIR)"/doctrees ./source  "$(TOXENVDIR)/build/html"
	
# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
