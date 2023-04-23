
.PHONY: *

# Admin
dev:
	pip install --progress-bar off -r requirements.txt >> log_install.txt
	python setup.py develop

lint:
	pycodestyle plotify/ --max-line-length=160

lint-examples:
	pycodestyle examples/ --max-line-length=80

lint-tests:
	pycodestyle tests/ --max-line-length=180

tests:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python -W ignore -m unittest discover -s 'tests' -p '*_test.py' -v
	make lint

notravis-tests:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python -W ignore -m unittest discover -s 'tests' -p '*_test_notravis.py' -v

alltests: 
	rm -f alltests.txt
	make tests >>alltests.txt 2>&1
	make notravis-tests >>alltests.txt 2>&1

predocs:
	cp ./README.md docs/index.md
	cp ./CHANGELOG.md docs/changelog.md

docs: predocs
	mkdocs serve -a 0.0.0.0:8000

docs-deploy: predocs
	mkdocs gh-deploy

# https://dev.to/neshaz/a-tutorial-for-tagging-releases-in-git-147e
release:
	echo 'Do not forget to bump the CHANGELOG.md'
	echo 'Tagging v'$(shell python -c 'print(open("plotify/_version.py").read()[15:-2])')
	sleep 3
	git tag -a v$(shell python -c 'print(open("plotify/_version.py").read()[15:-2])')
	git push origin --tags

publish:
	pip install -e .  # Full build
	rm -f plotify/*.so  # Remove .so files but leave .c files
	rm -f plotify/**/*.so
	python setup.py sdist  # Create package
	twine upload --repository-url https://upload.pypi.org/legacy/ dist/*  # Push to PyPI
