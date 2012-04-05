LATEX=pdflatex -interaction nonstopmode

%.pdf : %.tex
	mkdir -p build/
	${LATEX} -output-directory build/ $<
	-bibtex build/$(basename $<)
	${LATEX} -output-directory build/ $<
	${LATEX} -output-directory build/ $<
	mv build/$@ .

all: api.pdf
		 
clean:
	-rm -rf *.pdf build/
	
