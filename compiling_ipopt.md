# Compiling Ipopt with Mumps, HSL, and Pardiso

In order to achieve the best performance from the solver you should try different linear
solvers. For small problems it looks like mumps is good enough. For large problems, let us
say with more than 10 cities or regions you may need a better solver as MA97 or Pardiso.

This documents how I compiled Ipopt and then installed the compiled version to be used with
Ipopt.jl under Linux. This ia a small variation of [Ipopt's documentation](https://coin-or.github.io/Ipopt/INSTALL.html#COMPILEINSTALL).

## Obtain HSL and/or Pardiso 

You can follow Ipopt documentation and obtain HSL code with you are an academic, otherwise
it looks like you need to buy it. I am using the file coinhsl-2019.05.21.tar.gz.

Pardiso can be obtanined free for academic use in https://pardiso-project.org/ I got the
version `libpardiso600-GNU800-X86-64.so`. You should also add the lines below to your 
bashrc:

```bash
# Try to setup to run Ipopt in Julia with different linear solvers
export JULIA_IPOPT_LIBRARY_PATH=/usr/local/lib
export JULIA_IPOPT_EXECUTABLE_PATH=/usr/local/bin
export OMP_NUM_THREADS=<num. of cores>

# If you want to use Pardiso
export PARDISO_PATH=LOCATION_OF_YOUR_PARDISO_LIBRARY
export PARDISO_LIC_PATH=$PARDISO_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PARDISO_PATH
export PARDISOLICMESSAGE=1
```

## Setup

 Create a directory to compile Ipopt and friends. You will compile from there many
 packages. Also install some basic packages you will need:
 
 `sudo apt-get install gcc g++ gfortran git patch wget pkg-config liblapack-dev libmetis-dev openjdk11-jdk-headless`
 
Probably you also want to install openblas.

Below you will install libraries and binaries in `/usr/local` as root. You can uninstall
them using `sudo make uninstall`.

## Compile and Install ASL

You need to install ASL to get an executable that can be used with AMPL.

```bash
git clone https://github.com/coin-or-tools/ThirdParty-ASL.git
cd ThirdParty-ASL
./get.ASL
./configure
make
sudo make install
cd ..
```

## Compile and Install HSL

Start with

```bash
git clone https://github.com/coin-or-tools/ThirdParty-HSL.git
cd ThirdParty-HSL
```

Copy the HSL source (see above), untar it, and create a symbolic link named `coinhsl`
pointing to it. Then,

```bash
./configure
make
sudo make install
cd ..
```

## Compile and Install Mumps

```bash
git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git
cd ThirdParty-Mumps
./get.Mumps
./configure
make
sudo make install
cd ..
```

## Compile and Install Ipopt

It seems that if you link Pardiso and HSL at the same time you can get a core dump when
using HSL (probably due to Metis duplication at Pardiso binary). So if you plan to use
HSL, do delete the `--with-pardiso` option below.

```bash
git clone https://github.com/coin-or/Ipopt.git
cd Ipopt
export IPOPTDIR=$PWD
mkdir build
cd build
$IPOPTDIR/configure --with-asl --with-mumps --with-pardiso="$PARDISO_PATH/libpardiso600-GNU800-X86-64.so -fopenmp -lgfortran -llapack -lblas -lpardiso600-GNU800-X86-64 -L$PARDISO_PATH"
make
make test
sudo make install
cd ../..
```

## Build Ipopt.jl
Call the Julia REPL, rebuild `Ipopt.jl` and re-test it. 

```julia
import Pkg
Pkg.build("Ipopt")
# Create a small problem to test.
using JuMP
using Ipopt
m = Model(optimizer_with_attributes(Ipopt.Optimizer,
          "print_level" => 5, "linear_solver" => "ma57"))
@variable(m, x)
@objective(m, Min, (x - 0.5)^2)
optimize!(m)
```

If you get an error stating that `libhsl.so` is missing try to create a symbolic link
from `libcoinhsl.so` to `libhsl.so` in `/usr/local/lib`.

