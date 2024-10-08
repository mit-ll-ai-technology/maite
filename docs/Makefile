# Make cheat sheet
# := Simple assignment, evaluated once at initial assignment
# ?= Conditional assignment, assigns a value only if does not have a value
# $< First prerequisite
# $@ Target being generated

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=

# All build artifacts are held under BUILDIR,
# namely $(BUILDDIR)/source, $(BUILDIR)/html
BUILDDIR      ?= build

# SOURCEDIR is source directory given to sphinx. It holds all rst inputs,
# both those copied from docs/source and those generated from notebooks.
SOURCEDIR     := $(BUILDDIR)/source

# IPYNB_DIR holds jupyter notebooks that will be converted to .rst files
IPYNB_DIR := ../examples

# GENERATED_RST_DIR the location of the generated .rst files, same as the sphinx $(SOURCEDIR)
GENERATED_RST_DIR := $(SOURCEDIR)

IPYNB_PATTERN := $(IPYNB_DIR)/*/*.ipynb
IPYNB_FILES := $(wildcard $(IPYNB_PATTERN))

# GENERATED_RST_FILES the path of the .rst files generated from the $(IPYNB_DIR) files
GENERATED_RST_FILES := $(foreach file,$(IPYNB_FILES),$(GENERATED_RST_DIR)/$(subst $(IPYNB_DIR)/,,$(file:.ipynb=.rst)))

.PHONY: all clean notebook_rst html help

# First target is default
all: html

clean:
	@rm -r "$(BUILDDIR)"

$(GENERATED_RST_DIR)/%.rst: $(IPYNB_DIR)/%.ipynb
	python -m jupyter nbconvert --log-level=WARN --execute --to rst $< --output-dir $(abspath $(dir $@))

notebook_rst: $(GENERATED_RST_FILES)

html: notebook_rst
	cp -r ./source/* $(SOURCEDIR);
	# $GENERATED_RST_FILES already generated in $(SOURCEDIR)
	sphinx-build -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html



