EXTRA_OECMAKE:append = "\
    -D BUILD_opencv_gapi=OFF \
    -D WITH_TIMVX=ON \
    -D TIMVX_INSTALL_DIR=/usr/lib \
"

DEPENDS:append = " tim-vx"

INSANE_SKIP:${PN}-dbg += "libdir file-rdeps"
INSANE_SKIP:${PN} += "buildpaths"
