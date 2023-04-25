# IAQF2023

## Instructions
- Please make your own branch (use your name as the branch name) to work on the code.
- Only push to the main branch when you think it is something to share with the entire group (e.g. datasets, figures) or you are certain that the code is ready for production.
- Set up a virtual environment and keep track of your packages using the following commands.

### Clone the repository
````
    $ git clone https://github.com/joshuakim314/IAQF2030.git
````

### Git branch
````
    # to create a new branch
    $ git checkout -b ＜new-branch＞
    
    # to create a new branch based off of an existing branch
    git checkout -b ＜new-branch＞ ＜existing-branch＞
    
    # to switch branch (make sure to commit or push any untracked changes before switching to another branch)
    $ git checkout ＜branch-name＞
````

### Create a virtual env
````
    $ python3 -m venv venv
    
    # on mac terminal
    $ source venv/bin/activate

    # on windows in git bash
    $ source venv/Scripts/activate
````

### How to install a Python package using pip
````
    # to see your current package list
    $ pip3 list

    # to install a new Python package to your virtual env
    $ pip3 install <new-package-name>
````

### Pip freeze
````
    # shows a list of installed packages
    $ pip3 freeze
    
    # creates a file with a requirements list
    $ pip3 freeze > requirements.txt 
````

### Install dependencies
````
    $ pip3 install -r requirements.txt
````

### Push changes
````
    $ git add .
    $ git commit -m "write a meaningful commit message"
    $ git push
````
