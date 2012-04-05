all: configure
	cd src && ./waf.py build

configure:
	cd src && ./waf.py configure

clean: 
	rm -rf build/
