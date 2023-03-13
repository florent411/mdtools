import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='mdtools',
                version='0.0.1',
                author='Florent Smit',
                author_email='smitflorent@proton.me',
                description='A bunch of home-made tools used for the analysis of my MD and OPES simulations',
                long_description = long_description,
                long_description_content_type = "text/markdown",
                url="http://github.com/florent411/mdtools",
                license='None',
                classifiers = [
                    "Programming Language :: Python :: 3",
                    "License :: OSI Approved :: MIT License",
                    "Operating System :: OS Independent",
                ],
                package_dir = {"": "src"},
                packages = setuptools.find_packages(where="src"),
                python_requires = ">=3.6",
                install_requires= [
                    'numpy',
                    'seaborn',
                    'matplotlib',
                    'pandas',
                    'MDAnalysis',
                    'checkarg',
                    'torch',
                    'datetime',
                    'tqdm'
                ],
                zip_safe=False)