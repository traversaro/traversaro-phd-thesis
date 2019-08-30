# phd-thesis

Repo my PhD Thesis, available at https://traversaro.github.io/preprints/traversaro-phd-thesis.pdf .

For any doubt or to report an error on the thesis, please open an issue in this repo: https://github.com/traversaro/phd-thesis/issues/new .

The thesis uses the template available at https://github.com/kks32/phd-thesis-template .

## Compile the thesis 

### Dependencies (Debian/Ubuntu)
~~~
sudo apt install texlive-latex-base texlive-font-utils texlive-fonts-extra texlive-publishers texlive-science
~~~

### Generate PDF 
~~~
git clone https://github.com/traversaro/phd-thesis traversaro-phd-thesis
cd traversaro-phd-thesis/thesis
pdflatex thesis.tex
bibtex thesis.tex
pdflatex thesis.tex
pdflatex thesis.tex
~~~
