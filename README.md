# metafeatures-comparison
comparing the output for similar machine learning classifiers across various platforms

# Installation Instructions:
Create a directory in which to clone this repository.

command: mkdir metafeatures-comparison

Once inside the new directory, clone this repository.

command: git clone https://github.com/macetheace96/metafeatures-comparison.git

Then, create a virtual environment using the following command (you might have to install the virtualenv command first).

command: virtualenv -p python3 env

Then you must install dependencies for the python-weka-wrapper module using the following commands.

command: sudo apt-get install build-essential python3-dev

command: sudo apt-get install default-jdk

Now you can install the python dependencies listed in our requirements.txt file. First, though, you must activate your virtual environment.

command: source env/bin/activate

command: sudo env/bin/pip install -r metafeatures-comparison/requirements.txt

command: sudo env/bin/pip install -r metafeatures-comparison/weka_requirements.txt

Some files in this repository require matplotlib. However, matplotlib often does not work properly. In order to avoid this problem, we can link your computer's version of matplotlib to your virtual environment.

First, deactivate your virtual environment.

command: deactivate

Then, make sure that Tkinter is installed on your machine.

command: sudo apt-get install python3-tk

Next, install matplotlib for python3 using pip3 (you may have to install pip3 first).

command: sudo pip3 install matplotlib

Now, navigate to your virtual environment's python site-packages (the names of these directories might be slightly different depending on your version of python3).

command: cd env/lib/python3.5/site-packages/

Finally, we are going to link your computer's matplotlib with the virtual environment. The location of the packages listed might be slightly different on your computer (they might be in /usr/lib/ or /usr/local/lib). Furthermore, the names of the matplotlib packages might be slightly different. Just make sure you link all three matplotlib resources listed in your packages. Remember, these commands are given with the assumption that you have navigated to the site-packages directory of your virtual environment.

commmand: ln -s /usr/local/lib/python3.5/dist-packages/matplotlib .

commmand: ln -s /usr/local/lib/python3.5/dist-packages/matplotlib-1.5.1.egg-info .

commmand: ln -s /usr/local/lib/python3.5/dist-packages/matplotlib-1.5.1-nspkg.pth .

This concludes the installation portion.
