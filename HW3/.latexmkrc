# Make both `latexmk` and `latexmk -pdf` use XeLaTeX.
$pdf_mode = 1;
$pdflatex = 'xelatex -interaction=nonstopmode -halt-on-error %O %S';
