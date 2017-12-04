from distutils.core import setup, Extension

orca_module = Extension('orca',
                        sources = ['orca.cpp'],
                        extra_compile_args=['-std=c++11'],)

setup (name = 'orca',
       version = '1.0',
       description = 'ORCA motif counting package',
       ext_modules = [orca_module])

