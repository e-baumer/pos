from setuptools import setup, find_packages


setup(
    name='pos',
    version='0.1',
    description='Particle Swarm Optimization with Local Optimizers',
    url='http://github.com/e-baumer/pos',
    author='Eric Nussbaumer',
    author_email='ebaumer@gmail.com',
    license='GNU GPL',
    packages=find_packages(where='base'),
    zip_safe=False
)
