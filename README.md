# phd-thesis

The latest version of the thesis, automatically generated from the lastest commit on the `master` branch of the repo, is available at https://traversaro.github.io/phd-thesis/traversaro-phd-thesis.pdf .

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

## Acknowledgments  
* [Francesco Nori (`iron76`)](https://github.com/iron76)
* [Daniele Pucci (`DanielePucci`)](https://github.com/DanielePucci)
* [Andrea Del Prete (`andreadelprete`)](https://github.com/andreadelprete)
* [Alessandro Saccon](http://www.dct.tue.nl/asaccon/)
* [Vibhor Aggarwal (`vibhoraggarwal`)](https://github.com/vibhoraggarwal)
* [Prashanth Ramadoss (`prashanthr05`)](https://github.com/prashanthr05)
* [Lorenzo Rapetti (`lrapetti`)](https://github.com/lrapetti)
