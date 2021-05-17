import setuptools


setuptools.setup(
    name="peamt", # Replace with your own username
    version="0.2.0",
    author="Adrien Ycart",
    author_email="adrien.ycart@gmail.com",
    description="PEAMT: a Perceptual Evaluation metric for Automatic Music Transcription",
    long_description="""
    # PEAMT

    This package contains code to run PEAMT, a Perceptual Evaluation metric for Automatic Music Transcription.
    If you use any of this, please cite:

    Adrien Ycart, Lele Liu, Emmanouil Benetos and Marcus Pearce, 2020. ["Investigating the Perceptual Validity of Evaluation Metrics for Automatic Piano Music Transcription"](https://transactions.ismir.net/articles/10.5334/tismir.57), _Transactions of the International Society for Music Information Retrieval (TISMIR)_, 3(1), pp.68–81.

    ```
        @article{ycart2019PEAMT,
           Author = {Ycart, Adrien and Liu, Lele and Benetos, Emmanouil and Pearce, Marcus},
           Booktitle = {Transactions of the International Society for Music Information Retrieval (TISMIR)},
           Title = {Investigating the Perceptual Validity of Evaluation Metrics for Automatic Piano Music Transcription},
           Year = {2020},
           Volume = {3},
           Issue = {1},
           Pages = {68--81},
           DOI = {http://doi.org/10.5334/tismir.57},
        }
    ```

    For more info, please visit: [https://github.com/adrienycart/PEAMT](https://github.com/adrienycart/PEAMT)

    """,
    long_description_content_type="text/markdown",
    url="https://github.com/adrienycart/PEAMT",
    project_urls={
        "Bug Tracker": "https://github.com/adrienycart/PEAMT/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    install_requires=[
        'numpy',
        'pretty_midi==0.2.8',
        'mir_eval',
    ],

    package_dir={"": "peamt"},
    packages=setuptools.find_packages(where="peamt"),

    package_data={
        "peamt": ["model_parameters/*.pkl"],
    }
)
