import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='mdtools',
        version='0.1',
        description='A bunch of home-made tools used for the analysis of my MD and OPES simulations',
        url='http://github.com/florent411/mdtools',
        author='Florent Smit',
        author_email='smitflorent@proton.me',
        license='None',
        package_dir = {"": "src"},
        packages = setuptools.find_packages(where="src"),
        python_requires = ">=3.6",
        install_requires=['os', 'numpy', 'pandas', 'matplotlib', 'tqdm', 'time', 'MDAnalysis', 'checkarg'],
        zip_safe=False)