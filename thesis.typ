#set par(
  first-line-indent: 1em,
  justify: true,
)

#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#import "@preview/fontawesome:0.5.0": *

#show: codly-init.with()
#codly(languages: codly-languages + (
    dockerfile: (name: "Dockerfile", icon: [#fa-icon("docker") ], color: rgb("#0db7ed")),
    bitbake: (name: "BitBake", color: rgb("#0098db")),
    onnx-ir: (name: "ONNX IR", color: rgb("#323031"), icon: [#fa-icon("connectdevelop") ]),
), zebra-fill: rgb("#FFFFFF").darken(2%))

#import "@preview/glossy:0.8.0"
#let OpenVX = [OpenVX#sym.trademark]
#let VeriSilicon = [VeriSilicon]
#let NXP = [NXP]
#let pantone300 = rgb("#005BE8")
#let glossary = (
    nbg:
    (short: "NBG",
        description: "Network Binary Graph"),
    evk:
    (short: "EVK",
        long: "Evaluation Kit"),
    vsinpu:
    (short: "VsiNPU",
        long: "VeriSilicon™ NPU"),
    tim-vx:
    (short: "TIM-VX",
        long: "Tensor Interface Module for OpenVX"),
    xml:
    (short: "XML",
        long: "eXtensible Markup Language"),
    axi:
    (short: "AXI",
        long: "Advanced eXtensible Interface"),
    ahb:
    (short: "AHB",
        long: "Advanced High-performance Bus"),
    ssa:
    (short: "SSA",
        long: "Single Static Assignment"),
    tensor:
    (short: "tensor",
        description: "Sufficiently described for purposes of this text as multidimensional arrays"),
    IR:
    (short: "IR",
        long: "Intermediate Representation"),
    MAC:
    (short: "MAC",
        long: "Multiply–Accumulate Operation",
        description: [$a arrow.l a + (b times c)$]),
    HOF:
    (short: "HOF",
        long: "Higher Order Function"),
    ONNX:
    (short: "ONNX",
        long: "Open Neural Network Exchange"),
    TVM:
    (short: "TVM",
        long: "Tensor Virtual Machine"),
    
)

#show: glossy.init-glossary.with(glossary)

#import "@preview/clean-math-thesis:0.3.0": template

#show: template.with(
    title: "SW support for NPU accelerators for Linux-type operating systems",
    author: "Michal Žáček",
    supervisor1: "Petr Zemánek",
    supervisor2: "",
    program: "Systems and Virtualization",
    degree: "Bachelor's",
    university: "Czech Technical University in Prague",
    institute: "Faculty of Information Technology",
    deadline: "2025-05-26",
    city: "Prague",
    uni-logo: image("./logo_cvut_en.svg"),
    institute-logo: image("./information_technology.svg"),
    abstract: [
        == English

        === Abstract
        
        This document covers information about the
        NPU chip included in the NXP i.MX 8MP @evk,
        how it is used,
        what the different frontends to
        it have in common or differ in.
        We then perform some rudimentary
        benchmarks in order to
        determine the practical benefit of
        utilising the NPU for a common workload,
        before ending on a description
        of the components required for
        PikeOS to claim NPU support,
        and what porting them could entail.
        
        === Keywords
        hardware acceleration,
        embedded,
        machine learning hardware,
        linux,
        platform porting,
        TQ,
        NXP i.MX 8MP
        
        == Czech
        
        === Abstract
        
        Tento dokument zahrnuje informace o NPU
        čipu, kterým je vybavena deska NXP i.MX 8MP @evk,
        jak se používá,
        jaké k němu existují přistupové cesty,
        v čem se liší či co mají společného.
        Provedeme několik základních benchmarků
        abychom zjistili jaký je skutečný přínos
        chipu pro některé běžné ůkony.
        Nakonec shrneme co je potřeba aby PikeOS
        podporoval,
        pro oficiální podporu tohoto NPU čipu
        a probereme,
        co portování těchto knihoven může zahrnovat.
        
        === Keywords
        
        hardwarová akcelerace,
        embed,
        hardware na strojové učení,
        linux,
        porting mezi platformami,
        TQ,
        NXP i.MX 8MP
    ],
    cover-color: pantone300,
    heading-color: pantone300,
)

#show heading: it => {it; v(0.4em)}

= Declaration

I hereby declare that the presented thesis is my own work and that I have cited all sources of
information in accordance with the Guideline for adhering to ethical principles when elaborating an
academic final thesis.

I declare that I have not used any AI tools during the preparation and writing of the thesis. I am
aware of the consequences of apparently unacknowledged use of these tools in the production
of any part of my thesis.
Specifically the sole use was for ocassional primary source reconaissance,
akin to a search engine,
before processing and utilising the primary sources
on their own merit exclusively using my own eyes and words.

I acknowledge that my thesis is subject to the rights and obligations
stipulated by the Act No.,121/2000~Coll., the Copyright Act, as amended, in
particular that the Czech Technical University in Prague has the right to
conclude a license agreement on the utilization of this thesis as a school
work under the provisions of Article~60~(1) of the Act.


In Prague on May 16, 2025: #box(width: 1fr, repeat[.])

#import "@preview/muchpdf:0.1.1"

#let assignment = read("./zacekmi2-assignment.pdf", encoding: none)

#page(margin: 0pt, context{
    muchpdf.muchpdf(
        assignment,
        width: page.width,
        height: page.height,
    )
})

= Introduction

Our computing devices have come a long way in terms of speed,
and of course in terms of operating complexity.
This complexity can hold back very general purpose processing units
in performance when it comes to highly specialized tasks.
For a long time, GPUs were used where a large number
of specialized identical operations needed to be performed on
a vast array of differing data inputs,
as their specialized design for this use case made them incomparably faster than
using a common general-purpose CPU.
Though most often associated with their namesake of graphics,
they excel in many applications and have become a
valuable tool when working with neural networks.

Recently, even more specialized hardware has become available
to meet the rising demand to have access to these technologies
in more portable devices or for use in embedded situations.
Neural Processing Units#footnote[
Historically also called Versatile/Tensor/Intelligence Processing Units (VPU/TPU/IPU),
however for simplicity's sake we will use the term NPU and only deviate if a specific
function or product uses one of the alternatives
],
are becoming more common,
however these don't often come with universal drivers
and the most widespread standard today has no freely adoptable implementation.

Our goals in this thesis are to initially lay out the defined and known terms used in the context of neural nets, such as hardware, alternate names of known concepts,
or file formats.
Following this we shall go through the existing solutions implementing both the communication with actually hardware in the form of modules followed by frontend APIs for NPU work.
We will then go through the advantages that this hardware should in theory provide followed by a practical test of the actual results on real hardware.
The code will be made available so that it may be run to
test and/or verify these results.
Finally, we discuss what this means for PikeOS integration of NPU support.

As most of our sources are proprietary documentation,
this work will sadly be light on detailed images
and diagrams available in the primary sources.

= NPUs (Neural Processing Units)

Neural Processing Units (further shortened to NPU) are in general a type of hardware accelerator which is
optimized to efficiently handle Machine Learning workloads.
They offer both faster inference and often a lower energy footprint
on edge devices @BridgingQuantization2025.
Our NPU in question is composed of a frontend that is used for
communication between the NPU and the rest of the board,
a parallel processing unit,
a neural network engine
along with a backing storage.
The device works with 4-element vectors of 16/32-bit IEEE floating-point or 8/16/32-bit integers either signed or unsigned.
The device implements the #OpenVX hardware API
along with some extensions and
#VeriSilicon provides two separate official libraries through which to call this API.
The open-source @tim-vx @VerisiliconTim2025
which contains C++ bindings,
and Ovxlib which contains C bindings.
TIM-XV calls into one of two SDKs, the #VeriSilicon Unified #OpenVX SDK
supporting both compiling and running using either the GPU or NPU units,
and the
(as far as I can tell proprietary,
since the traces of this SDK online
are close to nonexistent)
VIP-Lite SDK that only contains the runtime for the NPU.
TIM-VX is used as the target for third-party frameworks targeting the board's NPU.
Of these we used specifically the former VeriSilicon SDK
also often listed as Vivante SDK or directly
`imx-gpu-viv`#footnote[The name is GPU, however it contains shared drivers for both NPU and GPU.]
in recipes,
as this was the one used by all of the NXP recipes.

The SDK includes the libraries
listed in @sdklist.
Of these only the ones marked with `*` are
linked against by #OpenVX.

#figure(
    image("./board-image.jpg"),
    caption: [The NXP i.MX 8MP @evk we worked on],
)<board_image>


#figure([
- `libArchModelSw.so*`
- `libCLC.so`
- `libGAL.so*`
- `libNNArchPerf.so*`
- `libOpenVX.so*`
- `libOpenVXC.so`
- `libOpenVXU.so`
- `libVSC.so*`
],
    caption: [List of SDK libraries])
<sdklist>

= File Formats

First, we present a rundown of the various
different formats encountered throughout this work,
commonly used in the machine learning space
and during yocto builds and flashes.

== Keras / HDF5

Keras is a library released in 2010
as part of a research effort,
originally at Alphabet @GoogleDevBlogChollet.
Its relation to TensorFlow will be apparent later.
Keras' currently used filetype `.keras` is a wrapper format consisting of
a ZIP-archive containing an `.h5` file
holding the layers and weights of a model,
and some `json` metadata about Keras' configuration,
the creation date, version information, etc.

HDF stands for Hierarchical Data Format;
it contains a POSIX-filesystem style hierarchy of groups
akin to directories,
even supporting hard and soft links.
Datasets take the role of files and contain
@tensor:pl with associated shapes and dtypes and
can be read and written to,
treating them as numpy arrays @HDF5PY,@HDF5FGGS.

#figure([
```python
import h5py

data = h5py.File("model.weights.h5")

# All valid
data['/layers/flatten/vars']
data['layers/flatten/vars']
subdata = data['layers']['flatten']
subdata['vars']

data['/layers/dense/vars/0']
#=> <HDF5 dataset "0": shape (784, 128), type "<f4">

data['/layers/dense/vars/0'][0]
#=> array([ 0.01777279,  0.07076738,
#          -0.05744237,  0.06602553,
#          ...])

```
],
    caption: [HDF5 structure access])
<hdf5>

As compression is done by Keras in-memory
saving and loading are transparent
to whether one is working with a ZIP archive
or a directory of files.

== Pickle

#{
set text(red)
[
    *This is dangerous to use.*
]
}

Some formats, especially those that allow a wide array
of Python datatypes, may allow unwanted code to execute as well
when loaded.
As a representative of these, we mention Pickle.
Since pickling allows arbitrary instances of classes,
malicious code may be inserted as well into any model downloaded.

An example taken from @ExploitingPythHamann2020
can be seen in @pickle_exploit.

#figure([
```python
import pickle
import os

class RCE:
    def __reduce__(self):
        cmd = ('echo GET HACKED')
        return os.system, (cmd,)


with open("tensor.bin", "wb") as f:
    pickle.dump(RCE(),f)
```],
    caption: [Class Exploiting Pickle deserialization],
)
<pickle_exploit>

== Block map `.bmap`

Relates to Yocto project's (formerly Intel's) `bmaptool` from @YoctoBmap,
which is used to flash data similarly to `dd`.
Unlike `dd`, this tool verifies the integrity
of flashed data, supports more complex arrangements
such as sourcing the image from a remote server,
supports sparse definitions,
and includes protections from accidentally destroying
data on disks that seem like regular mounted block devices.

== System Package Data Exchange `.spdx`

A Bill of Materials of all the included
packages used to build an image including all the versions
and metadata @SPDXDev.

== OpenEmbedded Image Creator `.wic`

Should your device require
multiple partitions @YoctoDevManual.
Use if `bmap` is not supported as it does not support
sparsness, etc.

== OpenEmbedded kickstart file `.wks`

Contains build commands for the `wic` command @YoctoDevManual.

== Flattened Device Tree `.fdt` / Device Tree Source `.dts`

Device tree source describes hardware @FreeBSD_FDT,
and is subsequently compiled into `.dtb`.
Determines how some things are flashed and written based on the target.

= #OpenVX & TIM-VX

#OpenVX is a standard hardware API that can be
implemented by hardware vendors of
hardware accelerators
and exposed as a C API to users @OpenvxPortab2011.
The Khronos group that
manages the #OpenVX specification
only describes an abstract machine
and certain operators with defined semantics.
The implementation of the operators
in terms of hardware
is completely up to the vendor,
and the specification aims to be written
in a way as to allow as much optimization
as possible.
When we spoke so far of `libOpenVX.so`,
that is #VeriSilicon's pre-compiled
implementation of the #OpenVX API.
#OpenVX officially aims at vision processing specifically,
however there are Neural Network extensions to allow
using the same pipeline to also accelerate
machine learning operations utilizing the @tensor
structure from the full #OpenVX v1.2 Spec,
and extending it with new operators such as
`vxActivationLayer`,
`vxConvolutionLayer`,
`vxFullyConnectedLayer`,
`vxSoftmaxLayer` and
more @OpenVXNNE.

Known Implementors of this API are:

/ VeriSilicon: The `libOpenVX.so` object all our libraries link against
/ KhronosGroup/OpenVX-sample-impl: #[
Only truly open-source implementation,
however, it is supposedly very ad-hoc,
slowly implemented,
and emulates the given operators on
CPU or related using OpenCL @KhronosgroupOp2024.
]
/ TexasInstruments/tiovx: Source available but only authorized for use on TI hardware.
/ AMD MIVisionX: #[
Part of the ROCm environment,
it implements #OpenVX over AMD's
GPUs and CPUS @OpenVXAMD.
]
/ Intel OpenVINO:
/ Nvidia VisionWorks:


However, only the relevant vendor makes sense
to use as it is tightly coupled to the given device
that is to be controlled with it.
The `x86_64` #OpenVX version
available from the TIM-VX repository
only emulates the behaviour of the API
using the machine's CPU.

#OpenVX still uses a graph abstraction;
however, the operations are far too low-level
to be convenient and useful for the purposes
of general-purpose machine learning jobs.
Especially since they do not
interoperate with existing well-known
formats from the public machine learning ecosystem.
Which is why we opted to rather
focus on relevant and convenient libraries
that wrap #OpenVX instead of calling it directly.

Further information about what the library
handles is given in the chapter on
Startup  (see @startup)).

#figure(
image("./timvx_overview.svg"),
    caption: [A diagram showing TIM-VX's role in running models on NPUs @VerisiliconTim2025])
<timvx_diagram>

TIM-VX is a C++ wrapper library meant for external
software vendors to link against
instead of #OpenVX for a more high-level
and universal API @VerisiliconTim2025.
As can be seen in @timvx_diagram) TIM-VX
is linked against by all our frameworks and many more,
including the Android soon to be deprecated NNAPI.
Some of these are officially supported
by the frameworks themselves, and
some have official NXP forks
or third-party forks.
Beyond the higher-abstractions specifically
tailored to machine learning, it also
includes many utilities for debugging.
The library additionally supports
boards other than our own.

= Setting up the environment

Our testing environment consists of a Yocto distribution,
running on our aformentioned chip.

`meta-overlay` will further be used to refer
to our custom layer for the purposes
of this thesis.

We use both the BSP and related layers directly from the TQ website,
followed by the #NXP meta-imx layer,
cloned via the `repo` @GoRepo utility from their manifest
repository @GithubNxpIm.

Repo is a utility developed by Alphabet
to work with multiple, completely separate
as far as Git is concerned,
repositories and treat them
as Git would treat submodules.
The repositories are initialized from
an @xml
file hosted in an arbitrary repository,
which contains a set of remotes and
"projects" (repositories)
that will be fetched into the final checkout.

#figure([
```bash
repo init \\
  -u https://github.com/nxp-imx/imx-manifest.git \\
  -b imx-linux-scarthgap \\
  -m imx-6.6.52-2.2.0.xml
```
],
    caption: [IMX repo initialization])
<repo>

Afterwards a bit of guesswork was required to get
all the compatible versions of all the layers and packages.
A combination of some layers from
the original BSP and
the NXP-IMX manifest was used in the end.
Mostly due to mismatches in `gstreamer` versions
as meta-imx requires a version not below 1.24.0,
but the bsp provides 1.22.5.
The result of this is found in @buildconf.

Following this we configure a build directory
with the Machine: `tqma8mpxl-mba8mpx`
and Distro: `fsl-imx-wayland`.

```bash
source setup-environment tqma8mpxl_build
```

Next we need an environment with all the programs that bitbake expects
and needs to function.
In our case we opted for an Ubuntu:22.04 Docker container,
with some extra packages installed as determined by consulting @YoctoPackages
initialized by @oci_init.

#figure([
```dockerfile
FROM ubuntu:22.04
RUN apt -o APT::Sandbox::User=root update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \\
      apt -o APT::Sandbox::User=root \\
      install -y gawk wget git diffstat unzip texinfo gcc \\
      build-essential chrpath socat cpio python3 python3-pip \\
      python3-pexpect xz-utils debianutils iputils-ping \\
      python3-git python3-jinja2 python3-subunit zstd \\
      liblz4-tool file locales libacl1
RUN locale-gen en_US.UTF-8
```
],
    caption: [Build Image Dockerfile])
<oci_init>

After building we may enter the environment
with @docker_init.

#figure([
```bash
docker run --rm -it \\
  -v .../scarthgap.TQ.ARM.BSP.0001:/src \\
  --userns=keep-id bitbake-env
```
],
    caption: [Entering Build Image])
<docker_init>

And we may now build the project:

```bitbake
bitbake imx-image-full
```

The output will be available under:

```
./tmp/work/imx8mpevk-poky-linux/imx-image-full/...
  .../1.0/deploy-imx-image-full-image-complete
```

This may take many hours.
Once successful, flashing is done with the command in @uuu_flash.

#figure([
```bash
sudo uuu -v -b sd_all \\
  ./images/imx-boot-tqma8mpxl-mba8mpxl-mfgtool.bin-flash_spl_uboot \\
  ./images/imx-image-full-tqma8mpxl-mba8mpxl.rootfs.wic
```
],
    caption: [Flashing image to device])
<uuu_flash>

== Kernel Tests

The VSOCK Makefile has no install goal, yet the recipe tries to call it,
so we just disable that part of the recipe with the following:

```bitbake
PACKAGECONFIG:kernel-tools = ""
```

== ONNXRuntime

We change the original ONNXRuntime recipe's
source from NXP to the official repo
as it now supports our NPU through the use of the
@tim-vx library (@onnx_build).

#figure([
```bitbake
ONNXRUNTIME_SRC ?= "gitsm://github.com/microsoft/onnxruntime.git"
SRC_URI = "${ONNXRUNTIME_SRC};nobranch=1;protocol=https
# Rel-1.21.0
SRCREV = "e0b66cad282043d4377cea5269083f17771b6dfc"
```
],
    caption: [ONNXRuntime Build recipe src patch])
<onnx_build>

ONNXRuntime's configuration script can't by default find our
installation of TIM-VX, so we must assist it a bit,
in addition to adding it to `(R)DEPENDS` (@onnx_depends).

#figure([
```bitbake
DEPENDS = "libpng zlib tim-vx tvm"
RDEPENDS:${PN} = "tim-vx tvm"

do_configure:prepend () {
    export TIM_VX_INSTALL="/usr"
}
```
],
    caption: [ONNXRuntime Build recipe dependency patch])
<onnx_depends>

Next we must also actually enable the use of this library
by adding the associated configuration flag
(@onnx_makeflags).

#figure([
```bitbake
EXTRA_OECMAKE += "\\
    -Donnxruntime_USE_VSINPU=ON \\
    -Donnxruntime_USE_TVM=ON \\
"
```
],
    caption: [ONNXRuntime Build recipe configure flags patch])
<onnx_makeflags>

As of writing the main `1.21.0` version of ONNXRuntime
is the first to support @vsinpu, however
the release version had broken support for
targets with no fp16 support,
therefore a pr's commit with the fix needs to be used.

`GitHub - Issue: 23957, PR: 23978`

It is very fast on track to be merged into main,
but now we apply these few commits with a patch.

```
SRC_URI:append = " file://fajin-corp_gh_pr_23978.patch"
```

Another patch is required though as the function

`VSINPUExecutionprovider::GetCapability`

in the file `vsinpu_execution_provider.cc`
calls a logger, yet omits to declare one so
we must add it manually.

```cpp
const auto& logger = *GetLogger();
```

The call to `save_build_and_package_info`
in `setup.py`,
causes an error.
It is only a buildinfo log,
so simply removing it creates a warning
when importing the module,
however without hindering further use.

The patch is included with the work.

```bitbake
SRC_URI:append = " file://fix_logger.patch"
```

== TIM-VX

This was simply needed to be updated to version 1.2.22,
from the official VeriSilicon repo @tim_bb.

#figure([
```bitbake
SRC_URI = "${TIM_VX_SRC};nobranch=1"
TIM_VX_SRC ?= "git://github.com/VeriSilicon/TIM-VX.git;protocol=https"
SRCREV = "8494275d7608942aa584c9c13bd5e2d77be9906c"
```
],
    caption: [@tim-vx's new version bb recipe])
<tim_bb>

== OpenCV

The build recipe for OpenCV is patchable,
we add TIM-VX to dependencies,
enable it in OpenCV and again
point the configuration script at our installation
directory with the file `opencv_\\%.bbappend`
containing @opencv_bb.

#figure([
```bitbake
EXTRA_OECMAKE:append = "\\
    -D BUILD_opencv_gapi=OFF \\
    -D WITH_TIMVX=ON \\
    -D TIMVX_INSTALL_DIR=/usr/lib \\
"

DEPENDS:append = " tim-vx"
RDEPENDS:${PN}:append = " tim-vx"

INSANE_SKIP:${PN}-dbg += "libdir file-rdeps"
INSANE_SKIP:${PN} += "buildpaths"
```
],
    caption: [OpenCV recipe to build with external TIM-VX])
<opencv_bb>

Compiling with external TIM-VX lib like this,
which is strongly advised against @OpenCVTimVxBackend,
yields a segfaulting library which can be seen in <cvvx_seg>.

#figure([
```python
>>> cv.setUseOpenVX(True)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
cv2.error: OpenCV(4.10.0) .../src/ovx.cpp:101:
  error: (-215:Assertion failed)
    !flag && "OpenVX support isn't enabled at compile time"
    in function 'setUseOpenVX'

>>> cv.useOpenVX()
False

# Segfaults
self._model = cv.FaceDetectorYN.create(...)
```
],
    caption: [Trying to enable OpenVX in OpenCV with external library])
<cvvx_seg>

Trying to compile TIM-VX directly in OpenCV requires us
to add Fortran to our toolchain due to a new dependence on
`lapack` lest we encounter error @fortran_missing_err.
We also must re-enable OpenCV downloads since the default
recipe tries to prevent random downloads during configuration,
by caching the downloads themselves.
This can be seen in our new recipe @cvvx.

#figure([
```
libgfortran was skipped:
  libgfortran needs fortran support to be enabled in the compiler
```
],
    caption: [Fortran missing from Toolchain error])
<fortran_missing_err>

#figure([
```bitbake
EXTRA_OECMAKE:append = "\\
    -D BUILD_opencv_gapi=OFF \\
    -D WITH_TIMVX=ON \\
    -D OPENCV_ALLOW_DOWNLOADS=ON \\
"

DEPENDS:append = " tim-vx lapack"
RDEPENDS:${PN}:append = " tim-vx lapack"
```
],
    caption: [OpenCV recipe to build with internal @tim-vx])
<cvvx>

Enabling Fortran breaks the `tpm2-tss-engine`
and `isp-imx-dev` packages
that is requires by `packagegroup-imx-ml`
so we try disabling what is wrong:

```bitbake
RDEPENDS:packagegroup-imx-ml:remove = " tpm2-tss-engine"
INSANE_SKIP:isp-imx-dev += "dev-elf"
```

This compiles an image,
however OpenCV still throws an error when calling
```python cv.setUserOpenVX(True)``` claiming
it is not compiled with the given support
and when run anyways with snippet @cv2py,
the speeds are identical for both
set backends,
however the speed seems suspiciously high
(in magnitude of hundreds of microseconds).

#figure([
```python
import cv2 as cv
m = cv.dnn.readNet('model.onnx')
m.setPreferable
m.setPreferableBackend(cv.dnn.DNN_BACKEND_TIMVX)
m.setPreferableTarget(cv.dnn.DNN_TARGET_NPU)
import numpy as np
m.setInput(np.full(fill_value=[1.0], shape=(1,28,28)))
m.forward()
```
],
    caption: [Attempt to run OpenCV DNN on NPU])
<cv2py>

= The Graph Workflow

Let us now describe how graphs are commonly constructed.

== Python Subclasses modelling @tensor:cap functions

All frameworks have Python APIs,
and all seem to use a similar form of abstraction.
They declare some class in the case of both PyTorch and @ONNX
this could be `torch.nn.Module` @ModulePytorc.
A model is then defined as a class that inherits from this root,
and overrides one or more specific methods.

Once we instantiate such a class
we receive an object that takes
@tensor:pl and returns results.
They can be freely composed into graphs
of arbitrarily complex modules containing modules
which all get compiled to a single graph.

For efficiency's and compatibility's sake these functions are
later compiled into other forms
especially when serialized into the form of any model file.

In the case of @ONNX specifically we may take
the example code @onnx_in.

#figure([
```python
class OnnxModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 3
```
],
    caption: [Python code to be converted to @ONNX @IR])
<onnx_in>

Which may be seen by the @ONNX exporter#footnote[Shortened by hand for readability] as @onnx_pseudo.

#figure([
```python
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[100, 128]"):
         # File: ./onnx.py:48 in forward, code: return x + 3
        scalar_tensor_default:
          "f32[]" = torch.ops.aten
                         .scalar_tensor
                         .default(3, dtype = torch.float32)
        add:
          "f32[100, 128]" = torch.ops.aten
                                 .add.Tensor(x, scalar_tensor_default)
        return (add,)
```
],
    caption: [@ONNX Pseudocode])
<onnx_pseudo>

Before being compiled into a low-level representation as shown in @onnx_ir.

#figure([
```onnx-ir
graph(
    name=main_graph,
    inputs=(
        %"x"<FLOAT,[100,128]>
    ),
    outputs=(
        %"add"<FLOAT,[100,128]>
    ),
) {
    0 |  # node_Constant_0
         %"val_0"<?,?> <- ::Constant() {
         %  value=Tensor<INT64,[]>(array(3), name=None)
         %}
    1 |  # node_Cast_1
         %"scalar_tensor_default"<FLOAT,[]> <- ::Cast(%"val_0") {
         %  to=FLOAT
         %}
    2 |  # node_Add_2
         %"add"<FLOAT,[100,128]> <- ::Add(%"x", %"scalar_tensor_default")
    return %"add"<FLOAT,[100,128]>
}
```
],
    caption: [@ONNX @IR])
<onnx_ir>

How this is done from arbitrary Python code is by utilising
a `FakeTensor` class,
on which the algorithm actually runs,
however instead of the operations
happening they are recorded in the background
and the result is again always a `FakeTensor`.
After the algorithm finishes,
the libraries inspect what operations were performed
on what @tensor:pl and constructs an equivalent program
through some backend, TorchScript, Dynamo, etc.
This approach allows quite arbitrary Python programs
to be compiled however at the loss of semantics.
For example the classic ```python map(lambda x: x + 2, list)```
would just be compiled into a series of completely unrelated
additions on some projected content of the @tensor.

These models also always have a static type of input and outputs
that may be unconstrained by the Python class which implements it
but must be specified explicitly when exporting.
The part of the type that determines the size of a @tensor
is usually called their "shape" in this context.
We set our input shape to `(100,128)` in this example
so the exported graph has annotations like `f32[100, 128]`.

Oftentimes a model is made to accept a large "batch" of inputs
to parallelize inference.
This is implemented as simply
setting the first dimension as dynamic,
resulting in shapes such as `(?,100,128)`.
Models must be explicitly resized before using this dynamic size
as it will default to 1 otherwise.

#figure([
```python
shape = (BATCH_SIZE, *shape[1:])
model.resize_tensor_input(0, shape)
model.allocate_tensors()
model.set_tensor(0, input_data)
```
],
    caption: [Setting input @tensor:pl with batches])
<py_batching>

#figure([
```python
[ sum(n) for n in x ]
```
],
    caption: [List comprehension in Python])
<py-list-comp>

The subset of python supported is quite large,
even code such as @py-list-comp,
will successfully compile into a graph,
the efficiency of which depends heavily on
the type of code,
for example the above generates a massive
array of round-robin addition nodes.
Of course calling external functions is also supported
and these are inlined and compiled as if written
directly inside forward.

If one wishes to have control over what
exactly their code compiles to,
it is better to directly use modules from the given library.

These can be initialized under `__init__`
as instance variables and then utilized in
the `forward` function like @py_super.

#figure([
```python
def __init__(self):
  super(..., self).__init__()
  self.relu = nn.ReLU()

def forward(self, x):
  x = self.relu(x)
  return x
```
],
    caption: [Python model module methods])
<py_super>

It is also useful to note that this code defines
string names for the inputs and outputs of a graph,
these are then used to retreive many outputs or set inputs
in a human-readable way.

=== `forward` (mandatory)

This is the crux of the pipeline,
it defines the actual operations performed on
the inputs and returns what the Graph node would have
as outputs.

If in need of multiple inputs and outputs
the function is allowed to take more than 1 argument,
and they will be matched according to name.
If on the other hand I want to return
multiple outputs we can wrap
the return in a dictionary with keys
being the names of our outputs and values being
the given output @tensor:pl.

=== `__init__`

Some frameworks, namely ONNXRuntime want their modules
to be attributes of the given object,
and so we usually then initialise our submodules here and refer to them in `forward`.
For example including `self.conv1 = ...` inside the constructor
and then calling as `x = self.relu(x)` in our `forward` function.

=== Movement functions

Methods such as `.to()`/`.cpu()`/`.ipu()`,
cause our model to be run on the given target device.
However, this may also hint to the given framework
what the target is which is also taken into account
in cases of optimization,
our chosen ones do not do this and use other ways to specify both
where and how to run.

== Exporting Models to files

This step is extremely important as it can make or break
the model's performance.
First and foremost, TensorFlow has the function
`tf.saved_model.save()`
which simply dumps a bunch of files into a directory
that represent the model.
This isn't as interesting as the `tf.lite.TFLiteConverter`
module.
The `converter` object is constructed from
one of a keras model,
a `saved_model` path generated by the above save function,
or a concrete function.
After creation we can set a vast number of options
such as `.optimizations`, `.inference_input_type`
or `.target_spec.supported_ops`.
Most importantly we should set optimizations to the value
`[tf.lite.Optimize.DEFAULT]`,
as without this the model won't quantize,
nor perform very well.
Once this is done setting inference in/out-put types
is optional and useful for quantization.
It must be stated however that
mixing types in the graph leads to something called
hybrid data types,
which are not supported by the NPU
and LiteRT will simply signal an error
and fall back onto the CPU delegate.

When quantizing the concept of a representative dataset comes into play.
In Python this takes the form of a generator
that returns a random assortment of inputs from a dataset,
so that the model may be inspected
for patterns,
minimum and maximum values that occur at different points,
in order to minimize the effect of quantization on accuracy.

== Keras

TensorFlow interoperates with and includes a complete interface to the Keras library,
which includes its own additional ways to define models.
For instance `.api.applications` under keras contains prebuilt models
for various uses such as image recognition
that can then be freely used and manipulated for our needs.
Another way is the
`.models.Sequential`
class,
which takes a list of layers and concatenates them
one after another into a model.
Keras layers are much like TensorFlow modules.

== Other Tools

=== Visualization: Netron & Zetane

Netron @NetronApp is an incredibly
useful open-source tool for inspecting,
visualizing and generally debugging machine learning model graphs.
Of relevant formats it supports @ONNX, TensorFlow and Keras,
along with support to display all metadata contained in a given layer.
It runs either as a web app or an offline local program.
Netron provides an interactive graph GUI
which allows searching, layouting, and
exporting to PNG or SVG.
An example of such an export in @netron.

#figure(
    image("./netron_test.svg"),
    caption: [Example of a Netron graph render])
<netron>

Zetane @Zetane deserves an honourable mention,
it is a closed-source batteries included
environment for designing ML software and
includes a viewer very similar to 
Netron.
This viewer however is much
more capable and includes 3D visualizations
of convolutions on images, heatmaps,
graphs of values,
running the model with weights
and inspecting propagation.
It is also callable directly via a Python API,
so unlike Netron it can be integrated as a frontend into
other projects.

=== Conversion: Tensor2onnx & Tflite2onnx

All manner of programs exist to convert between
model formats,
which is not a trivial task however.
Very generally models can be converted
but some programs lose details,
such as weights, names,
structures that are concisely represented
may unwrap into larger forms, and more.
For a given application,
care should be given to verify
that all required information is intact.

= Hardware Specifics

This entire section references mainly @IMx8mPlusAp,VivanteVIP.
Most importantly the NPU implements the #OpenVX API,
and so can be controlled using the #VeriSilicon `libOpenVX` implementation.

== NPU Contents

Let us now walk through the contents of our NPU chip.
Very little information is available publicly.

=== Parallel Processing Unit (PPU)

SIMD4, with 4 units with 256 threads each.
General purpose highly-parallel unit for standard
arithmetic operations.

=== Neural Network Engine

Specialized support for convolutions.
Performs 1152 @MAC operations per clock cycle.

=== @tensor:cap Processor

Supported operations are listed in @opst.
Handles operations other than convolutions.
Does not seem to support 32 bit floats.

#figure(table(columns: 2,
    [Pooling], [Max, average],
    [Unpooling], [Yes],
    [Activation], [ReLU, Leaky ReLU (LUT for other types)],
    [Normalization], [Yes],
    [Region Proposal Support], [Yes]
),
    caption: [@tensor:cap Processor supported operations])
<opst>

It is perhaps useful to mention that some interfacing is proprietary such as interrupts.
The NPU can send CPU interrupts, these are set using the driver and cannot in fact be understood without the driver since the NPU and driver both assign arbitrary meaning to the given bits of the interrupt information register.

== Configuration Environment Variables read by @tim-vx

In the i.MX Machine Learning User's Guide @IMxMachineLeLfRe2024 section 6.1.2,
we can read about configuration variables.

#figure(table(columns: 2,
    [`USE_GPU_INFERENCE`], [As the NPU and GPU share this driver,
        TIM-VX uses the value of this variable to determine
        which to use, `"1"` for GPU or `"0"` for NPU],
    [`CNN_PERF`],
    [Prints how long operations take.
        Requires `VIV_VX_DEBUG_LEVEL=1` and implied by `VIV_VX_PROFILE`],
    [`NN_EXT_SHOW_PERF`],
    [Shows the details on how the compiler determines performance],
    [`VIV_VX_PROFILE`],
    [Enables creation of `vprofiler_xxx.vpd` files which may be examined using the Vivante vAnalyzer tool from the Vivante VDK.
        Information is either per-node (value: `"1"`) or per-graph (value: `"2"`)],
    [`VIV_VX_DEBUG_LEVEL`],
    [Prints extra debug information],
    [`VIV_MEMORY_PROFILE`],
    [Only applies to CPU/GPU],
    [`VIV_VX_ENABLE_CACHE_GRAPH_BINARY`],
    [Enables saving of the compiled graph to disk with a hash
        so that the next time the same graph would be compiled,
        this `*.nb` file will be loaded instead.
        In addition to this the documentations claims that
        warmup time may take more than one inference],
    [`VIV_VX_CACHE_BINARY_GRAPH_DIR`],
    [Directory to save the cache files to],
),
    caption: [Configuration Environment Variables read by @tim-vx]
)
<model_caching>

== Power Modes

The NPU can be set to 4 different states,

#figure(table(columns: 2,
    [On], [Standard full-power],
    [Off], [Can be powered off, since after leaving this state the device is reinitialized],
    [Idle], [Clock speed lowered to $1/64^"th"$],
    [Suspend], [Idle clock speed and requires some time to reach idle],
),
    caption: [Power Modes],
)


= Frameworks

We will now cover the most popular and best supported
frameworks for the NPU.

== LiteRT (Lite RunTime)

This project is closely related to TensorFlow
and acts as its ambassador for resource-constrained devices.
Tensorflow is used to train, prepare, and debug the model,
before it's deployed to a device
where only a stripped down runtime is installed.
As such it does not contain much beyond what is needed
to run inference.
Models need to be converted
into the FlatBuffers tflite format before use @LitertOverview.

When running the library we must set an environment variable which `libvx` checks
to see which device it should use, either NPU or GPU:

```bash
export USE_GPU_INFERENCE=0
```

Next we must load the external dynamic library
which we pass into the LiteRT interpreter in @litert_delegate.

#figure([
```python
import tflite_runtime.interpreter as tflite

external_delegates = [
    tflite.load_delegate("/usr/lib/libvx_delegate.so", "")
]
```
],
    caption: [Loading LiteRT delegate])
<litert_delegate>

Delegates are shims between the Tensorflow python library
and the external system library acting as device driver,
so even though we want to use `libOpenVX.so`,
we instead load the `libvx_delegate.so`
library which links against it.

#figure([
```python
interpreter = tflite.Interpreter(
  model_path=args.model_file,
  experimental_delegates=external_delegates,
)
```
],
    caption: [Create the LiteRT interpreter object])
<litert_inter>

After creating the interpreter object with @litert_inter
we can interact with it in one of two ways.
Either we have a prepared calling convention called a signature
inside the model and call that as in @litert_signat
which directly returns an output @tensor.

#figure([
```python
signature = model.get_signature_runner()
signature(x=<tensor>)
```
],
    caption: [LiteRT signature runner])
<litert_signat>

Or we go about explicitly setting and reading
each input and output.
For this it is useful to know their properties,
so we may call ```python model.get_input_details()```
or ```python model.get_output_details()``` respectively.
These functions return lists of detail objects
containing the shape, name and index.
This index can be used to read or write @tensor:pl
with ```python set_tensor``` in @litert_set_tensor.

#figure([
```python
model.set_tensor(model.get_input_details()[0]['index'], input_data)
model.get_tensor(model.get_output_details()[0]['index'])
```
],
    caption: [LiteRT directly setting input @tensor:pl])
<litert_set_tensor>

== @TVM:both

As the full name Apache @TVM @ExtendingTensoWang2022 might suggest,
this is an open-source framework built by Apache
in a similar vein to @ONNX including their own
@IR called Relay that is supposed to allow
optimizations to be shared between multiple target backends.

It does not have its own format for files,
opting to be able to import any of the other frameworks'.
It does have the additional capability to compile
down models into C++ libraries in the form
of shared object files.

According to the recipe's `EXTRA_OECMAKE` and the fact that it links against tim-vx
in the tvm bitbake recipe,
support for @vsinpu should already be built in.

The ```python tvm.relay.frontend``` module allows us to load
a model, from almost any format user in machine learning,
using functions such as ```python from_keras()```, ```python from_onnx()```,
```python from_paddle()``` or ```python from_tflite()```.

However merely trying to import the ```python tvm.relay``` module
throws an error concerning python not being able to find
the scipy library.
Scipy is not packaged by any of the standard NXP layers
and didn't have a workable recipe that could be found
before this went out of scope for this work.

Another interesting property of @TVM is that
@ONNX can use it as a backend directly.
Strangely enough though @ONNX even when compiled with
```bash -Donnxruntime_USE_TVM=ON```
does not acknowledge @TVM as a valid backend.

== @ONNX:both

Though originally authored by joint efforts of Facebook and Microsoft,
this project has flourished into a
widely supported open-source ecosystem @ONNXWeb.
It comprises a comprehensive specification for models, formats, types, operators and
abstract data descriptions.
It includes protobuf definitions of their `.onnx` model files.
It used to support only inference,
but training was added with @ONNX @IR spec version 7 @OpenNeuralNet.
Models here are represented as either just a stateless
inference function in the case of inference-only models,
or may be extended with an initialization and a training
method which may modify internal stateful variables
of the given model.
@ONNX also encodes a block of operators that encode a more complex task,
these may be substituted for builtins by the runtime based off
of their name#footnote[Went through changes in IRv9].
The internal structure is a list of acyclically dependent topologically sorted nodes,
each node providing a name, metadata, i/o and the given operation performed.
@ONNX however does provide #ref(<HOF:short>)-like operators that are applied
to entire subgraphs,
thus substituting the need for self-references @OnnxConcepts.
These nodes are strung together as a pipeline each input being connected
to a previously declared output of the same name.
Each output name is unique since @ssa is mandatory,
verification tools for this and other properties are
available from the creators of @ONNX.
The last argument of some operators is marked as variadic,
thus allowing as is traditional with
regular languages to pass an arbitrary number of
inputs/outputs to it,
obviously respecting the minimum arity.
These graphs are meant to be assembled programatically,
however @ONNX does provide a textual form of their files.

Implementing a backend for @ONNX is
done by providing a Python shim wrapping the functionality
as we have compiled in support for @vsinpu,
we can simply list it as our inference session provider
and run inference as is shown in @onnx_infer.

If ONNXruntime is downloaded from pip,
we can use the available provider
CPU and Azure.
@vsinpu is listed as known but unavailable.
Activating it yields an unavailable error
so the @ONNX we use must be from Yocto.

#figure([
```python
import onnxruntime
session = onnxruntime.InferenceSession(
  "model.onnx",
  providers = [ "VSINPUExecutionProvider" ]
)
session.run(
  input_feed= {
    "x": np.full(
           fill_value=[1.0],
           shape=(1,28,28),
           dtype=np.float32,
         )
  },
  output_names = ["add"],
)
outputs = session.run(None, {"input": inputTensor})
```
],
    caption: [Running @ONNX on NPU])
<onnx_infer>

You may convert any torch model into @ONNX.
@torch2onnx shows how
given a torch/tensorflow model one can export
it into an onnx file.

#figure([
```python
torch.onnx.export(
  model, # model being run
  torch.randn(1, 28, 28), # model input (or a tuple for multiple inputs)
  "fashion_mnist_model.onnx", # where to save the model
  input_names = ['input'], # the model's input names
  output_names = ['output'], # the model's output names
)
```
],
    caption: [Exporting Torch to @ONNX])
<torch2onnx>

== Unaddressed frameworks

=== The NXP eIQ environment

The NXP eIQ stack officially supports, LiteRT, @ONNX, PyTorch and OpenCV.
Of these only LiteRT officially supports the included NPU.
Due to the fact that it only calls the other libraries listed
and that the environment itself
is a large and unwieldy project,
we opted to skip it in this work.

=== Android NNAPI

Android provides a direct official API
for use in Machine Learning acceleration @AndroidNNAPI.
It is however only exclusively available on Android
which requires negotiating with NXP
for an image and is to be deprecated in the future
in favour of LiteRT.
The Linux Kernel also has something called the
NNAPI, however that is completely unrelated.

=== Paddle Paddle

A popular Chinese framework for machine vision,
it seems well-documented but much of the
community and so documentation is
mandarin @GithubPaddle.
It is not mentioned by NXP in documentation,
however it is marked as having supported
for @tim-vx, so may be explored in the future.

= Performance

The entire point of the NPU is that
the specialized hardware therein
should perform these specific following
tasks very swiftly and efficiently.
We will now discuss what affects the final performance
and then put this claim to a practical test.

=== Startup
<startup>

This is apparent when running the C++ example where
measuring the entire program's runtime
results in 0.2 seconds on CPU and
3.4 seconds when running on NPU
even though the internal timer
measuring pure inference time shows
that NPU speeds up inference from 35ms to 3ms.

After first starting a program you must run
at least one so-called "warmup" inference
which leads to the pipeline of
compilation, optimization,
deciding on tiling,
and generally preparing the model for a run,
before actually running inference on the given model.
We state "at least one" because
that is what the documentation states,
we only encountered warmup requiring one
inference run,
however documentation advises to first
test how warmup behaves for the given model
and then determine how to benchmark.
This takes time in a magnitude of seconds to
tens of seconds depending on model size.
After this first inference all subsequent inferences
should be a consistent and much higher speed,
as long as that model's handle is kept @AN12964.
Multiple models can be kept in memory
at once,
each needs its own warmup phase and then
as long as they fit,
they can be kept and used at will
intermittently.

Some frameworks offer a way to compile
multiple models so they efficiently live together
in memory @CoralEdgeTpuCompiler.
This might of course cause some models
to have most of the cache memory to themselves,
while leaving the rest with very little,
however it prioritizes the models
based on the user's preference
so it is a tradeoff of note
having to expulse the cache every time
a model is switched and may at times be faster
even considering that.

The abstract graph built by the user
is analyzed by the implementation
and operations may be merged,
changed and transformed
as to provide the best possible
performance @OpenvxGraphOpAbeysi2019.
This is what takes up
the crux of the startup delay
and what graph binary caching (see @model_caching))
tries to address.
It does not affect runs of instance of a program,
but rather skips part of the "warmup"
inference in subsequent runs.

Input data must be split into
smaller fixed-size chunks to fit
onto on-chip memory and
so that possible offloads into
DRAM and other expensive
memory operations may be minimized.
This is done automatically by the implementation
however shows up often in performance logs.
The DMA controller then requests
and processes tiles as the NPU requires.

=== Quantization

Since the NPU itself is built for
all of 8/16/32-bit wide floats @IMxMachineLeLfRe2024
and different sizes of integers,
we may wish to lessen the load and speed-up the inference
by opting to sacrifice the precision of the result
and instead of running the entire process
with F32, we use INT8,
according to @PyTorchQuant
this may lead to the operations being implemented 2 to 4
times faster with integers.

=== Dynamic loading of Libraries

In the case of both the `Python` and `C++` interfaces,
using either `ONNX` or `LiteRT`,
the case is always that the interfacing library
must be loaded dynamically.
Loading may take up a significant amount of time,
in one experiment the Python profiler
shows 7\\% of the runtime taken up by `do_lookup_x` from `ld-linux.so`.

=== Bus

Since the NPU acts as a completely separate device from the
standard SOC's CPU and memory,
even having its own clock,
we must transfer the model and input parameters
over an @axi/@ahb bus
and that takes time.

Behind that Bus lies a Memory Controller,
scheduler and more which all work together to slow down the call
especially initially.

If a @tensor or other piece of our
pipeline is marked as being bound
to memory on the device it is
called a "device tensor" @DeviceTensors.
As an example CUDA allows
the creation of a batch of memory marked
"CudaPinned" which is asynchronously
accessible to the device.

=== NPU Clogging

During tests we often observed that if the input data
is too large, or the NPU is otherwise mishandled
then all further requests to it simply
block forever.
I am sure the NPU can be reset,
however we opted to just restart the device for now
whenever this occurs,
sometimes having to hard-kill all processes that
are working with the NPU as they even block system restarts.

== Benchmarking Practical Results

Running the same model under LiteRT and @ONNX,
yields interestingly different results for CPU
speed and NPU Warmup, which makes sense
as even though the models implement the same
structure they have different representations and
so may be compiled very differently.
Seems that once the model is up and running though
the NPU has the same performance under both.

#figure(table(columns: 4,
    [Framework], [CPU], [NPU Warmup], [NPU],
    [ONNX], [50ms], [16ms], [4ms],
    [LiteRT], [132ms], [380ms], [2.3ms],
    ),
    caption: [mobilenet_v1_1.0_224_quant Performance Metrics])
<qperf>

With this we can estimate what this NPU would allow us to do
in terms of live image classification.
Even though the input image is only 224 by 224 pixels,
using the CPU we would only get 7 images identified per second,
while using the NPU that goes up to 435 images per second,
for example we could classify in real time every single
frame of 18 different cameras before dropping
below 24 frames per second on each.

#figure(table(columns: 4,
    [Type], [CPU], [NPU Warmup], [NPU],
    [F32], [107ms], [6.4s], [11ms],
    [I8], [107ms], [6.5s], [10ms],
    [F32 (No optimization)], [98ms], [749ms], [345ms]),
    caption: [MobileNetV3Large Quantization tests])

@qperf shows us that the biggest difference
lies in whether or not optimizations are enabled.
For the next @vgg19perf
we use the built-in `keras.applications`
models.
They are ready-made, of various sizes,
well-documented,
don't use exotic operators,
and accurately depict what a common
workload might look like,
so we decided they'd make good sources of
performance data.
Other ML papers on the matter
seem to also use some combinations
of these model architectures to
show results
@BridgingQuantization2025.

#figure(table(columns: 4,
    [Model], [CPU], [NPU Warmup], [NPU],
    [VGG19], [4s], [37s], [34ms],
    [MobileNetV3Large], [107ms], [6.4s], [11ms],
    [MobileNetV3Small], [36ms], [2.6s], [4ms],
    [MobileNetV2], [82ms], [5.1s], [9ms],
    [MobileNet], [131ms], [5.5s], [7ms]),
    caption: [Various Models performance])
<vgg19perf>

As the NPU is optimized only for specific tasks
so certain operands may end up just being run
on the CPU even if the NPU is set as the target,
only emitting a Warning notice to standard error.

In addition, an inefficiently built model may even run slower
on NPU due to the various overheads if
it is not something that gets optimized,
bumping a 52ms runtime on the CPU up to
a consistent 200ms on the NPU.
I reiterate,
enabling hardware acceleration slowed down
our inference to 25% of what it achieved on just CPU.

Testing with absolutely minimal models like the one
suggested for usage on the MNIST dataset,
shows that below some threshold of size
the CPU is again faster.
For example the NPU will not go below 400μs
while the CPU manages 180μs.

= Suggesting PikeOS integration

We see that the most integral part of
porting this ecosystem to a given OS,
would be to get TIM-VX working as
that is targeted by all the other
libraries.
For that,
the following must first be addressed.

/ Porting libOpenVX: #[This will be the most difficult part
as that is closed-source and shipped as a prebuilt
binary artefact.
This shared object binary can of course be patched
and edited to work around some issues;
however, there is little that can be done
if the library does not work
    for some other more nuanced reason.]
/ Runtime dependencies: #[
Luckily, libOpenVX has very few runtime dependencies.
Apart from the standard set of linux libraries
libOpenVX requires libVSC, libGAL, libArchModelSw and libNNArchPerf,
all of which are also shipped as binary blobs.
Only shipping libOpenVX and omitting TIM-VX
does not lessen the load substantially,
    as all of these dependencies are also required by it.]
/ Compile TIM-VX against our libOpenVX library and ship it: #[
All our mentioned frameworks have multiple sets of bindings,
primarily always Python and C++, along with a mix of other languages.
For performance sensitive applications, it would be beneficial
to support the C++ API while Python will depend on
whether clients wish to prototype their applications
on the device or if the workflow
of training on external hardware and only running on the device
is sufficient.
]
/ Wrapper ML frameworks need to be ported: #[
Once that is in place @TVM and OpenCV support extrernal
file types,
while @ONNX, Keras and TensorFlow
provide tooling to convert and
often have built-in facilities to work with,
import and export
all of their respective formats.
And so supporting one of these well should
suffice for most applications.
We would suggest the Lite Runtime as that is the one "blessed"
by NXP as their primary supported framework
and it has the largest community.
]

= Conclusion

We set out to perform a reconaissance of
hardware acceleration
in the context of machine learning on embedded devices.
To determine how the hardware is accessed,
what external libraries let us interface into it,
and to test that the performance gains are as claimed and suspected.

Much of the authors' time was spent
on familiarization with the bitbake Yocto build system,
BSPs,
flashing to the @evk,
navigating hardware vendor documentation,
and other themes completely tangential
to the problems we set out to address.
Despite that, we identified the primary
point of ingress for controlling the NPU,
several libraries that support it
even outside the NXP specification.

We either compiled these libraries with NPU support,
recorded what steps were done to do so,
and afterwards described how to specify to the library
that it should utilize the NPU.
Or in the cases where a roadblock was hit,
we described what the next step would
be in getting it to work.
All the code used is included
as part of this work.

We identified a good set of models to use for benchmarks
and then benchmarked the libraries on CPU and NPU,
concluding that in terms of raw NPU
inference perforance on the same model
their differences are negligible,
if optimized correctly.
And that the the NPU itself achieves
tenfold or hundredfold performance compared to the CPU,
when configured correctly,
depending on model.

Finally, we outlined which libraries are relevant
for PikeOS to claim NPU support
and briefly discussed from an outside perspective
how that might be done.

= Build configuration
<buildconf>

```
Build Configuration:
BB_VERSION           = "2.8.0"
BUILD_SYS            = "x86_64-linux"
NATIVELSBSTRING      = "universal"
TARGET_SYS           = "aarch64-poky-linux"
MACHINE              = "tqma8mpxl-mba8mpxl"
DISTRO               = "fsl-imx-wayland"
DISTRO_VERSION       = "6.6-scarthgap"
TUNE_FEATURES        = "aarch64 armv8a crc crypto"
TARGET_FPU           = ""
meta
meta-poky            = "HEAD:200d12b6a58ad961d60a7774ca0f7a9d29498724"
meta-oe
meta-python
meta-multimedia      = "HEAD:72018ca1b1a471226917e8246e8bbf9a374ccf97"
meta-freescale       = "HEAD:0627128b341cfb2bef7a0832ce8cac0ce1127f13"
meta-qt6             = "HEAD:586a6cb5aec755803a3be3cec359baafe89d6432"
meta-tq              = "HEAD:257b8c0b4b6df3bb27fb69bd2312dd254c73fed3"
meta-imx-ml
meta-imx-sdk
meta-imx-bsp         = "HEAD:219f6d04a4c339eb6f2dc626f944bbdf9a716ff5"
meta-arm
meta-arm-toolchain   = "HEAD:950a4afce46a359def2958bd9ae33fc08ff9bb0d"
meta-freescale-distro = "HEAD:b9d6a5d9931922558046d230c1f5f4ef6ee72345"
meta-overlay         = "<unknown>:<unknown>"
meta-virtualization  = "HEAD:6f3c1d8f90947408a6587be222fec575a1ca5195"
meta-filesystems
meta-networking      = "HEAD:72018ca1b1a471226917e8246e8bbf9a374ccf97"
meta-tpm
meta-parsec          = "HEAD:459d837338ca230254baa2994f870bf6eb9d0139"
meta-clang           = "HEAD:2b7433611d80f6d0ee1b04156fa91fc73d3c2665"
```

#glossy.glossary()

#let nocite(label) = cite(label, form: none)

#nocite(<HowToUseOpen>)
#nocite(<HttpsWwwTq>)
#nocite(<YoctoDevManual>)
#nocite(<YoctoPackages>)
#nocite(<IMx8mPlusDataSheet>)

#bibliography("./thesis.bib", style: "iso-690-numeric")

