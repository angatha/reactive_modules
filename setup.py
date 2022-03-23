from setuptools import setup

install_requires = [
    'tabulate>=0.8.9',
]

setup(
    name='reactive_modules',
    version='1.0.0',
    packages=['reactive_modules'],
    url='https://github.com/angatha/reactive_modules',
    entry_points={"console_scripts": ["reactive_modules = reactive_modules.cli:main"]},
    install_requires=install_requires,
    license='MIT',
    python_requires=">=3.10",
    description='Libary to work with reactive modules',
)
