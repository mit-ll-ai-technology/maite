# Building the Docs

## Via tox

From `maite/` run

```shell
> pip install tox
> tox -e docs
```

The HTML files will be saved to `/.tox/docs/build/html`.

## Manually

Install `maite` with doc-building requirements:

```shell
> pip install ".[builddocs]"
> cd docs
docs> pip install -r requirements.txt
```

Then build the docs:
 
```shell script
docs> python -m sphinx source build
```

The resulting HTML files will be in `maite/docs/build`.