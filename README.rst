f2f-fitting
===========

.. image:: https://readthedocs.org/projects/f2f-fitting/badge/?version=latest
	:target: http://f2f-fitting.readthedocs.io/en/latest/?badge=latest
	:alt: Documentation Status

This is a Python package to fit 3D morphable models (3DMMs) to images and depth maps of faces. It mainly provides classes to work with and render 3DMMs, and functions that use these classes to optimize the objective of fitting 3DMMs to a source RGB image or depth map. The package is tentatively named ``mm``, so to use it, follow the installation instructions below and import it like you would any other Python package. ::

	import mm
	mm.do_stuff()

Features
========

* Fit a 3DMM shape model to a source depth map.
* Process 3DMM shape fittings from the frames of a source video containing a person speaking to find a new sequence of 3DMMs to match a target speech audio file.
* Fit a 3DMM texture model with spherical harmonic lighting to a source RGB image.
* Recover the barycentric parameters of the underlying verticles from the 3DMM mesh triangles that contribute to each pixel of a person's face in an image.

The project is still under development, and the big missing feature to be implemented is:

* Fit a 3DMM shape model to a source RGB image.

For more info, check out the `documentation on Read the Docs <http://f2f-fitting.readthedocs.io/en/latest/>`_.

Prerequisites
=============

* Python 3
* For the package alone, you need: PyOpenGL, numpy, scipy, librosa, scikit-learn
* For the scripts in ``bin/``, you may also need: matplotlib, scikit-image, mayavi, h5py, hmmlearn, networkx, and
* `Volumetric Regression Network (VRN) <https://github.com/AaronJackson/vrn>`_

	* This is to generate a volume of a person's face from an image, and we fit the 3DMM shape model to this volume. It is the workaround until we implement 3DMM shape model fitting to an image.

Installation and Development
============================

Because the project is still under development, you are encouraged to fork the repository on GitHub and install the package to your local Python environment using ``pip install -e <repository-root-directory>``. Then you can develop the package and open a pull request to suggest your improvements to the package. For example, using Terminal:

1. Clone your fork onto your computer: ``git clone https://github.com/<your-username>/f2f-fitting.git``.
2. Change directories to the root of the forked repository: ``cd f2f-fitting``.
3. Add the original repository as a remote: ``git remote add upstream https://github.com/ids-cv/f2f-fitting.git``.

	* This will allow you to sync your fork with updates made to the original repository, as outlined later below.

4. Install via ``pip``: ``pip install -e .``.

	* This creates a symbolic link to the package in your Python path so that you can use features from the package while being able to develop simultaneously without having to continually uninstall and reinstall the package.

Here is an example workflow:

1. Get the most recent updates from the original repository: ``git fetch upstream``.
2. Make sure you are in your fork's ``master`` branch: ``git checkout master``.
3. Merge the changes in the original repository to your fork: ``git merge upstream/master``.
4. Create a new branch on your Github fork for an issue or improvement that you want to work on: ``git checkout -b <your-branch-name>``.
5. Work on the issue, test your solution, and if you are satisfied, commit your changes to your new branch: ``git commit -am "<your-commit-message>"``.
6. Merge your changes to your fork's ``master`` branch. ::

	git checkout master
	git merge <your-branch-name>

7. Delete your branch, now that you are done with the changes you wanted to make: ``git branch -d <your-branch-name>``.
8. Go on the GitHub website of the original repository and open a pull request to have your changes incorporated in the original repository.

Support
=======

If you are having issues, please let us know. You can also email me: leonnguyen94@gmail.com.
