test:
	py.test --doctest-modules

install:
	python setup.py install

reinstall:
	pip uninstall glia -y
	python setup.py install
