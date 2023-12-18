# Building the Docs

## Via tox

From `maite/` run

```shell
tox -e docs
```

The HTML files will be saved to `/.tox/docs/build/html`.

## Manually

Install `maite` and install the docs requirements:

```shell
/maite/docs> pip install -r requirements.txt
```

Then build the docs:
 
```shell script
/maite/docs> python -m sphinx source build
```

The resulting HTML files will be in `maite/docs/build`.

## Converting IPython Notebooks

To convert an .ipynb notebook in the examples folder into a markdown file suitable for the docs, navigate to the examples folder containing the notebook, then run the command:

```shell
jupyter nbconvert --to markdown your_notebook.ipynb
```

This will create a markdown version of the notebook in the examples folder.  Then, cut and paste it to the relevant docs folder.