EXTRA_OECMAKE:append = "\
    -D BUILD_opencv_gapi=OFF \
    -D WITH_TIMVX=ON \
    -D OPENCV_ALLOW_DOWNLOADS=ON \
"

DEPENDS:append = " tim-vx lapack"
RDEPENDS:${PN}:append = " tim-vx lapack"

INSANE_SKIP:${PN}-dbg += "libdir file-rdeps"
INSANE_SKIP:${PN} += "buildpaths"

