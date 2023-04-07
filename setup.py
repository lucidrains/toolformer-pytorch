from setuptools import setup, find_packages

setup(
  name = 'toolformer-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.20',
  license='MIT',
  description = 'Toolformer - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/toolformer-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'automated-tool-use'
  ],
  install_requires=[
    'beartype',
    'einops>=0.4',
    'torch>=1.6',
    'tqdm',
    'x-clip'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
