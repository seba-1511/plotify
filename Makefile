
all: plot

install:
	python setup.py develop

plot: 
	python tests/plotting.py

anim: 
	python tests/animation.py

publish:
	#http://peterdowns.com/posts/first-time-with-pypi.html
	# TODO: Version bump (2x setup.py) + GH Tag release
	# git tag 0.1 -m "Adds a tag so that we can put this on PyPI."
	# git push --tags origin master
	# python setup.py register -r pypi
	# python setup.py sdist upload -r pypi
	python setup.py sdist  # Create package
	twine upload --repository-url https://upload.pypi.org/legacy/ dist/*  # Push to PyPI
