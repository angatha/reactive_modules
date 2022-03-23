# reactive_modules

1. [Installation](#installation)
2. [Using the cli](#using-the-cli)

## Installation

Having the environment active into which you want to install this module, execute the following depending on what your
use case is:

The latest version package can be directly installed from github
with `pip install https://github.com/angatha/reactive_modules/archive/master.zip`.

If you want to modify the source code of reactive_modules, `pip -e <path to local copy>` is preferred.

After this, the package is available in your environment. In addition, the `reactive_modules` cli is also available
inside the environment. Now you can run e.g. `reactive_modules -h`.

## Using the cli

After installing the package, inside the environment you will have a new command: `reactive_modules`. It provides three
features from the modules to the commandline:

1. formatting code (`reactive_modules format ...`)
2. checking the syntax and types (`reactive_modules check ...`)
3. run one execution path of one module (`reactive_modules run ...`)

The next step to learn the cli would be to run `reactive_modules -h` to see the help and options for the different sub
commands. Some examples are given in the `examples` directory.