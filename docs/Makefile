# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
# ('make' without arguments defaults to first target)
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean notebook_markdown

clean: 
	@rm -r "$(BUILDDIR)"

# Make cheat sheet
# $< First prerequisite
# $@ Target being generated

# Make sure the .ipynb files do trigger the catch-all target
%.ipynb:

source/explanation/protocol_overview.md: ../examples/protocol_overview.ipynb
	jupyter nbconvert --to markdown $< --output-dir $(dir $@)

source/tutorials/torchvision_object_detection.md: ../examples/torchvision_object_detection/torchvision_object_detection.ipynb
	jupyter nbconvert --to markdown $< --output-dir $(dir $@)

notebook_markdown: source/explanation/protocol_overview.md source/tutorials/torchvision_object_detection.md

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
