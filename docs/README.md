# Building the Docs

## Via tox

From `jatic-toolbox/` run

```shell
tox -e docs
```

The html will be saved to `/.tox/docs/build/html`.

## Manually

Install `jatic-toolbox` and install the docs-requirements:

```shell
/jatic/docs> pip install -r requirements.txt
```

Then build the docs:
 
```shell script
/jatic/docs> python -m sphinx source build
```

The resulting HTML files will be in `jatic-toolbox/docs/build`.
