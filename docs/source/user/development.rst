Development
===========

The ``sst`` toolkit is developed by Martin Thoma, Vitali Kaiser, Marvin
Teichman, Sebastian Bittel. The development began in June 2015.

Its base git repository is at
``/disk/no_backup/mlprak1/git-repos/team-segmentation``.

You can send me an email: info@martin-thoma.de

This package needs nolearn. Nolearn is a wrapper for Lasagne. Lasagne makes use
of Theano to compute symbolic expressions.


Getting started
---------------

Clone the repository:

.. code:: bash

    your/home/dir$ git clone /disk/no_backup/mlprak1/git-repos/team-segmentation


Install the minimum dependencies:

.. code:: bash

    your/team-segmentation/dir$ pip install -r requirements.txt


Install everything which is still missing. You can use

.. code:: bash

    $ sst --selfcheck

to find out which Python packages are still missing.



Tools
-----

* ``nosetests`` for unit testing
* ``pylint`` to find code smug
* git for code and documentation versioning


Code coverage can be tested with

.. code:: bash

    $ nosetests --with-coverage --cover-erase --cover-package sst --logging-level=INFO --cover-html


You can add a so called pre-commit-hook to git to run all tests automatically
before you commit. The hook is installed by copying the following code to
`.git/hooks/pre-commit` and making this file executable:

.. code:: bash

    #!/usr/bin/env bash

    # Stash uncommited changes to make sure the commited ones will work
    git stash -q --keep-index
    # Run tests
    nosetests sst/
    RETURN_CODE=$?
    # Pop stashed changes
    git stash pop -q

    exit $RETURN_CODE


Documentation
-------------

The documentation is generated with `Sphinx <http://sphinx-doc.org/latest/index.html>`_.
On Debian derivates it can be installed with

.. code:: bash

    $ sudo apt-get install python-sphinx
    $ sudo -H pip install numpydoc

Sphinx makes use of `reStructured Text <http://openalea.gforge.inria.fr/doc/openalea/doc/_build/html/source/sphinx/rest_syntax.html>`_

The documentation can be built with ``make html``.

The documentation is written in numpydoc syntax. Information about numpydoc
can be found at the `numpydoc repository <https://github.com/numpy/numpydoc>`_,
especially `A Guide to NumPy/SciPy Documentation <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.
