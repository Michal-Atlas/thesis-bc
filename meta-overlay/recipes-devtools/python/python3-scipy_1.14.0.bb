SUMMARY = "SciPy: Scientific Library for Python"
HOMEPAGE = "https://www.scipy.org"
LICENSE = "BSD-3-Clause"
LIC_FILES_CHKSUM = "file://LICENSE.txt;md5=3387f60dd97ca5d8ea2a907e50aec3a0"

inherit pkgconfig pypi python_mesonpy

# SRC_URI[sha256sum] = "cd58a314d92838f7e6f755c8a2167ead4f27e1fd5c1251fd54289569ef3495ec"
SRC_URI[sha256sum] = "b5923f48cb840380f9854339176ef21763118a7300a88203ccd0bdd26e58527b"

DEPENDS += " \
	${PYTHON_PN}-numpy-native \
	${PYTHON_PN}-pybind11-native \
	${PYTHON_PN}-pythran-native \
	${PYTHON_PN}-gast-native \
	${PYTHON_PN}-beniget-native \
	${PYTHON_PN}-ply-native \
	lapack \
"

DEPENDS:append:class-target = " \
	${PYTHON_PN}-numpy \
"

RDEPENDS:${PN} += " \
	${PYTHON_PN}-numpy \
"

PACKAGECONFIG ?= "lapack"

PACKAGECONFIG[openblas] = "-Dblas=openblas -Dlapack=openblas,,openblas,openblas"
PACKAGECONFIG[lapack] = "-Dblas=lapack -Dlapack=lapack,,lapack,lapack"
PACKAGECONFIG[f77] = "-Duse-g77-abi=true,,,"

CLEANBROKEN = "1"

export LAPACK = "${STAGING_LIBDIR}"
export BLAS = "${STAGING_LIBDIR}"

F90:class-native = "${FC}"
F90:class-target = "${TARGET_PREFIX}gfortran"
export F90

export F77 = "${TARGET_PREFIX}gfortran"

BBCLASSEXTEND = "native nativesdk"
