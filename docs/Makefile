# Makefile for building Sphinx documentation

SPHINXBUILD = sphinx-build
SOURCEDIR = source
BUILDDIR = _build

.PHONY: help html clean

help:
	@echo "Makefile for Sphinx documentation"
	@echo ""
	@echo "Available targets:"
	@echo "  html    Build HTML documentation"
	@echo "  clean   Remove build artifacts"

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html
	@echo "Documentation built in docs/_build/html."

clean:
	rm -rf $(BUILDDIR)
	@echo "Cleaned docs build artifacts."
