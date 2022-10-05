# pypi.org


# test.pypi.org

twine upload --repository testpypi dist/*

# RTZR Internal

```bash
VERSION=0.0.7
python setup.py sdist bdist_wheel --universal
twine upload dist/* -r rtzr
git tag -a -f -m '' v${VERSION}
git push -v origin refs/tags/v${VERSION}
```