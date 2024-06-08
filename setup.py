from setuptools import setup, find_packages

setup(
    name='flashembed', 
    version='0.0.1', 
    packages=find_packages(),
    install_requires=[
        'tokenizers',
        'onnxruntime',
        'numpy',
        'huggingface_hub',
        'tqdm',
        'llama-cpp-python==0.2.67'
    ],  
    author='Prithivi Da',
    author_email='',
    description='Lightweight & Fast Python library to add low-footprint (all-MiniLM-* equivalent) multilingual retrievers to your RAG and Search & Retrieval pipelines.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/PrithivirajDamodaran/flashembed',  
    license='Apache 2.0',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
