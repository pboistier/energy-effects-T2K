Usage
=====

Installation
------------

To use this project on your machine, you can copy the repository:

.. tabs::

   .. tab:: GitHub-CLI

      .. code-block:: console

         gh repo clone pboistier/energy-effects-T2K
   
   .. tab:: SSH

      .. code-block:: console

         git clone git@github.com:pboistier/energy-effects-T2K.git

   .. tab:: HTTPS

      .. code-block:: console

         git clone https://github.com/pboistier/energy-effects-T2K.git

Then, you'll need pip and pdm:

.. code-block:: console

   python -m pip install --upgrade pip
   pip install pdm

Inside the local project directory, run:

.. code-block:: console

   pdm install

Finally activate the virtual environment using:

.. code-block:: console

   eval $(pdm venv activate)

That's it! You can fully run all the project in the venv.

To deactivate the venv, simply run:

.. code-block:: console

   pdm deactivate