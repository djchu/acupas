## DESCRIPTION
This package implements faster cutting algorithms for binary and multiclass linear SVM proposed in the following paper:

Dejun Chu, Changshui Zhang, Qing Tao, A faster cutting plane algorithm with accelerated line search for linear SVM, Pattern Recognition (2017), http://dx.doi.org/10.1016/j.patcog.2017.02.006

## Compilation
On linux system, type 'make' to build the 'acupa' for binary linear SVM and 'acupam' for multiclass SVM.

## Usage
Syntax for binary case:
acupa [options] training_set_file testing_set_file

For multiclass case:
acupam [options] training_set_file testing_set_file

where data files are stored in SVM^{light} format.

This code is based on [LIBOCAS](http://cmp.felk.cvut.cz/~xfrancv/ocas/html/) tools. So the usage is similar to LIBOCAS in options, in which '-m 1' chooses OCAS solver while '-m 2' chooses ACUPA solver.

## Contact
This package was developed by Dejun Chu (djun.chu@gmail.com). Due to limited number of tests performed, some bugs may still remain. Any suggestions, bug reports and improvements are welcome.

If this implementation helps your research, please cite the paper above.
