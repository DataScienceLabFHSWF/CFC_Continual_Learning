import os
import sys
sys.path.insert(0, os.path.abspath('../../mammoth'))

project = 'CfC Continual Learning'
author = 'Felix Neubürger'
release = '0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'alabaster'
html_static_path = ['_static']
html_theme_options = {
    'description': 'Documentation for the CfC continual learning project',
    'show_related': True,
}
