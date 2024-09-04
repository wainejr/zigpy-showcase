import os
import shutil
import pathlib
import subprocess
import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize


zig_args = "--release=off"

def build_lib(path: str):
    old_path = os.getcwd()
    os.chdir(path)
    subprocess.check_call(f"{sys.executable} -m ziglang build {zig_args}".split(" "))
    os.chdir(old_path)


def copy_lib(path_zig: pathlib.Path, path_lib: pathlib.Path):
    extensions = [".so*", ".dll*", ".dylib*"]
    for ext in extensions:
        path_so = path_zig.glob(f"zig-out/lib/*{ext}")
        path_lib.mkdir(exist_ok=True, parents=True)
        # Copy each .so file
        for lib in path_so:
            shutil.copy(lib, path_lib)


def copy_include(path_zig: pathlib.Path, path_inc: pathlib.Path):
    path_inc = path_zig / "zig-out/include/"
    path_inc.mkdir(exist_ok=True, parents=True)
    shutil.copytree(path_inc, path_inc, dirs_exist_ok=True)


def build_zig():
    zigpath = pathlib.Path(__file__).parent / "zigpy/mzig"
    path_lib = pathlib.Path(__file__).parent / "zigpy/mzig/lib"
    shutil.rmtree(path_lib, ignore_errors=True)
    # shutil.rmtree(path_inc, ignore_errors=True)

    build_lib(zigpath.as_posix())
    copy_lib(zigpath, path_lib)
    # copy_include(zigpath, path_inc)


def build(setup_kwargs):
    build_zig()

    # from https://stackoverflow.com/questions/63679315/how-to-use-cython-with-poetry
    # setup_kwargs.update({
    #         'ext_modules': cythonize(
    #             extensions,
    #             language_level=3,
    #             compiler_directives={'linetrace': True},
    #         ),
    #         'cmdclass': {'build_ext': build_ext}
    #     })

    # from https://stackoverflow.com/questions/16993927/using-cython-to-link-python-to-a-shared-library
    # build "myext.so" python extension to be added to "PYTHONPATH" afterwards...
    libs_names = ["octree"]
    extensions = []
    for lib_name in libs_names:
        ext = Extension(
            f"zigpy.mzig.{lib_name}",
            sources=[
                f"zigpy/mzig/{lib_name}.pyx",
                # "SomeAdditionalCppClass1.cpp",
                # "SomeAdditionalCppClass2.cpp"
            ],
            include_dirs=["zigpy/mzig/src/"],
            library_dirs=["zigpy/mzig/lib/"],
            runtime_library_dirs=["zigpy/mzig/lib/"],
            libraries=[f"{lib_name}"],  # refers to "lib{lib_name}.so"
            # cython_directives = {'language_level': "3str"},
            # extra_compile_args=["-O3"],
        )
        extensions.append(ext)

    setup_kwargs["ext_modules"] = cythonize(extensions, compiler_directives={"language_level": "3str"}, force=True)
