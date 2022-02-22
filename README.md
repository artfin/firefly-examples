# Basic Quantum Chemistry calculations 

Do some basic quantum chemistry with the help of [Firefly](http://classic.chem.msu.su/gran/firefly/index.html) (version 8) computational chemistry program developed at MSU by Alex A. Granovsky.

Making Vim recognize the Firefly input filetype `.fly`:
*  `mkdir --parents ~/.vim/ftdetect; echo "au BufRead,BufNewFile *.fly set filetype=gamess" > $_/gamess.vim`
*  `mkdir --parents ~/.vim/syntax; cp editor/gamess.vim $_` 
