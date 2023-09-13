# glacier-bay-uwt-student-projects: UWT Student Glacier-to-Bay Sediment Transport Projects

This is the online location for data and data processing notebooks for the glacier-to-bay sediment tracking projects at UWT. Data will also be archived on Teams.

## Notation
In the instructions below, `code` formatting refers to filenames, keys to press, or something you need to type. Bracketed italics like <code>[<i>this</i>]</code> refer to text that you'll need to substitute with something appropriate to your situation (your username, a name you choose for your branch, the name of a package you want to install, etc.).

## Prerequisites

To prepare to use the software and data in this folder, you will need to install the following:
- [Anaconda Python](https://www.anaconda.com/download) - This distribution of the Python programming language comes with most of the packages (like plugins in other software) you will need. Use the free version, since you are not using it for commercial work.
- Also install the following packages using <pre><code>conda install [<i>package_name</i>]</code></pre>:
  - Altair: [altair](https://anaconda.org/conda-forge/altair) - Data visualization package, like ggplot in R
  - Altair Data Server: [altair_data_server](https://anaconda.org/conda-forge/altair_data_server) - For visualizing large datasets
- Follow the instructions on the [PmagPy site](https://pmagpy.github.io/PmagPy-docs/installation/PmagPy_install.html) to install the PmagPy package for processing paleomagnetic and rock magnetic data. *NOTE*: follow the detailed instructions under "You can find more specific pip install instructions" for your particular system. *Do not*  do the `pip install` commands listed on the main page.
- If you are doing image analysis, you will need [Fiji](https://imagej.net/software/fiji/) or another [ImageJ](https://imagej.net/software/imagej2/) implementation.
- _OPTIONAL_: If you are interested in contributing to the data and notebooks here, you will probably want to learn about the `git` version control system. It's sort of like google docs, but for code (and a lot harder to use...). If you are interested, download [git](https://git-scm.com/) and read [this introduction to using git and github](https://www.digitalocean.com/community/tutorials/how-to-create-a-pull-request-on-github#create-pull-request). 

## Downloading
Most students will want to follow the "Easy" directions below.

### Easy: Zip File
The easiest way to get the data and notebooks from this site is to click the "<> Code" button on the home page, and select "Download Zip" from the menu that pops up. This will save a compressed (.zip) file to your computer's drive. Navigate to the file and use whatever tools your system has to extract it to a stable place in your computer's file system (I store mine in a folder called "Research" in my Documents; you should do whatever makes sense to you). You should end up with a folder called `glacier-bay-uwt-student-projects` with a number of files, folders, and subfolders.

### Difficult: Git
This page is the home page of a set of online files and folders on the Github website called a _repository_. The `git` software allows you to put a version of this repository on your computer's file system (called your _local_ repository), make changes, and then upload the changed version to Github (called the _remote_ repository) without overwriting other users' changes. We can then decide as a group which changes to accept and which will just remain your own, on your local system. Do do this, you need to create your own version of the repository online, called a _fork_.

1. Create a [Github](https://github.com/) account for yourself, and make sure you have `git` installed on your computer.
2. From the main page of this repository, click the "Fork" button. Using the _Owner_ menu, choose yourself as the owner. Write a description of what you plan to do, and click "Create fork".
3. Once the new fork is created, click the name of the repository on the screen that comes next. Make sure you are viewing the fork you just created: your username will appear in the top left of the page, near the Github icon. If it doesn't, click the triangle next to the Fork button and select your username.
4. Click the green "<> Code" button and copy the URL displayed under "Code".
5. Open up a Powershell (Windows) or Terminal (Mac) window and change directory (<pre><code>cd <i>[path_to_folder]</i></code></pre>) to the place in your file system where you want to store your work.
6. Type the command <pre><code>git clone <i>[copied_url]</i></code></pre>, where _`copied_url`_ is the URL you copied in step 4.
7. Change your working folder into the folder you just created (probably `cd glacier-bay-uwt-student-projects`). This is called the _base directory_ for the repository.
8. Type the command <pre><code>git branch <i>[new_branch]</i></code></pre>, where _`new_branch`_ is a title that describes some change in the code or data that you are working on (e.g. `process-irm-data` or `image-analysis`). Titles can't have spaces in their names (we usually use dashes instead), and are usually all lower case.
9. Type the command <pre><code>git checkout <i>[new_branch]</i></code></pre>.
10. Make any changes to the files or folders you need to make (using whatever software you want).
11. When you are done making changes, go back to Powershell or Terminal and navigate back to the base directory.
12. Type the command `git add \*` or `git add -A` to _get your local system ready_ to sync all of your changes with the remote repository.
13. Type the command <pre><code>git commit -m "<i>[message]</i>"</code></pre>, where _`message`_ is a description of the changes you made.
14. Type the command <pre><code>git push --set-upstream origin <i>[new_branch]</i></code></pre> to _actually_ sync the changes with the remote repository.

 ## Contents
 This repository is organized as follows:
 ```
.
├── README.md
├── data
│   ├── images
│   ├── processed
│   └── raw
└── notebooks
```
- `README`: This file.
- `data`: Folders containing raw or processed data, or images derived from data (i.e. visualizations of data).
  -   `images`:  Charts, graphs, images, or other visualizations of data
  -   `processed`: Data that have been processed (e.g. hysteresis statistics, image analysis results)
  -   `raw`: Data that come directly from an instrument or have been typed in from a lab notebook (e.g. specimen masses, hysteresis measurements). Contains sub-folders for each instrument.
-   `notebooks`: Folder containing Jupyter notebooks (`.ipynb` files) used for processing, analyzing, or visualizing data, along with other Python code (`.py` files).

 ## Running Notebooks
 Once you have Anaconda installed, you'll need to start the "Jupyter Notebook" application from your system's Start/Windows or Apple menu. That will open a web browser window that you should use to navigate to the notebook you want to run. Double click the file name to access the notebook. Click the arrow button next to each code block to run that code block (or click in the block and press `Control + Enter` or `Shift + Enter`). [This blog post](https://www.dataquest.io/blog/jupyter-notebook-tutorial/) has additional helpful background.
