# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/GGNNC++

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/GGNNC++

# Include any dependencies generated for this target.
include Matrix/CMakeFiles/common.dir/depend.make

# Include the progress variables for this target.
include Matrix/CMakeFiles/common.dir/progress.make

# Include the compile flags for this target's objects.
include Matrix/CMakeFiles/common.dir/flags.make

Matrix/CMakeFiles/common.dir/matrix.cpp.o: Matrix/CMakeFiles/common.dir/flags.make
Matrix/CMakeFiles/common.dir/matrix.cpp.o: Matrix/matrix.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/GGNNC++/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Matrix/CMakeFiles/common.dir/matrix.cpp.o"
	cd /root/GGNNC++/Matrix && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/common.dir/matrix.cpp.o -c /root/GGNNC++/Matrix/matrix.cpp

Matrix/CMakeFiles/common.dir/matrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/common.dir/matrix.cpp.i"
	cd /root/GGNNC++/Matrix && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/GGNNC++/Matrix/matrix.cpp > CMakeFiles/common.dir/matrix.cpp.i

Matrix/CMakeFiles/common.dir/matrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/common.dir/matrix.cpp.s"
	cd /root/GGNNC++/Matrix && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/GGNNC++/Matrix/matrix.cpp -o CMakeFiles/common.dir/matrix.cpp.s

Matrix/CMakeFiles/common.dir/matrix.cpp.o.requires:

.PHONY : Matrix/CMakeFiles/common.dir/matrix.cpp.o.requires

Matrix/CMakeFiles/common.dir/matrix.cpp.o.provides: Matrix/CMakeFiles/common.dir/matrix.cpp.o.requires
	$(MAKE) -f Matrix/CMakeFiles/common.dir/build.make Matrix/CMakeFiles/common.dir/matrix.cpp.o.provides.build
.PHONY : Matrix/CMakeFiles/common.dir/matrix.cpp.o.provides

Matrix/CMakeFiles/common.dir/matrix.cpp.o.provides.build: Matrix/CMakeFiles/common.dir/matrix.cpp.o


# Object files for target common
common_OBJECTS = \
"CMakeFiles/common.dir/matrix.cpp.o"

# External object files for target common
common_EXTERNAL_OBJECTS =

Matrix/libcommon.so: Matrix/CMakeFiles/common.dir/matrix.cpp.o
Matrix/libcommon.so: Matrix/CMakeFiles/common.dir/build.make
Matrix/libcommon.so: /usr/local/blas/OpenBLAS/lib/libopenblas.a
Matrix/libcommon.so: /usr/local/blas/CBLAS/lib/cblas_LINUX.a
Matrix/libcommon.so: Matrix/CMakeFiles/common.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/GGNNC++/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libcommon.so"
	cd /root/GGNNC++/Matrix && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/common.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Matrix/CMakeFiles/common.dir/build: Matrix/libcommon.so

.PHONY : Matrix/CMakeFiles/common.dir/build

Matrix/CMakeFiles/common.dir/requires: Matrix/CMakeFiles/common.dir/matrix.cpp.o.requires

.PHONY : Matrix/CMakeFiles/common.dir/requires

Matrix/CMakeFiles/common.dir/clean:
	cd /root/GGNNC++/Matrix && $(CMAKE_COMMAND) -P CMakeFiles/common.dir/cmake_clean.cmake
.PHONY : Matrix/CMakeFiles/common.dir/clean

Matrix/CMakeFiles/common.dir/depend:
	cd /root/GGNNC++ && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/GGNNC++ /root/GGNNC++/Matrix /root/GGNNC++ /root/GGNNC++/Matrix /root/GGNNC++/Matrix/CMakeFiles/common.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Matrix/CMakeFiles/common.dir/depend

