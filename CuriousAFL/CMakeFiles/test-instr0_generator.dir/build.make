# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

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
CMAKE_COMMAND = /opt/clion-2019.2.1/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /opt/clion-2019.2.1/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/david/Documents/gits/CuriousAFL/CuriousAFL

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/david/Documents/gits/CuriousAFL/CuriousAFL

# Utility rule file for test-instr0_generator.

# Include the progress variables for this target.
include CMakeFiles/test-instr0_generator.dir/progress.make

CMakeFiles/test-instr0_generator: .test-instr0


.test-instr0:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/david/Documents/gits/CuriousAFL/CuriousAFL/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating: /home/david/Documents/gits/CuriousAFL/CuriousAFL/.test-instr0"
	/opt/clion-2019.2.1/bin/cmake/linux/bin/cmake -E echo 0 | /home/david/Documents/gits/CuriousAFL/CuriousAFL/afl-showmap -m none -q -o /home/david/Documents/gits/CuriousAFL/CuriousAFL/.test-instr0 /home/david/Documents/gits/CuriousAFL/CuriousAFL/test-instr

test-instr0_generator: CMakeFiles/test-instr0_generator
test-instr0_generator: .test-instr0
test-instr0_generator: CMakeFiles/test-instr0_generator.dir/build.make

.PHONY : test-instr0_generator

# Rule to build all files generated by this target.
CMakeFiles/test-instr0_generator.dir/build: test-instr0_generator

.PHONY : CMakeFiles/test-instr0_generator.dir/build

CMakeFiles/test-instr0_generator.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test-instr0_generator.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test-instr0_generator.dir/clean

CMakeFiles/test-instr0_generator.dir/depend:
	cd /home/david/Documents/gits/CuriousAFL/CuriousAFL && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/david/Documents/gits/CuriousAFL/CuriousAFL /home/david/Documents/gits/CuriousAFL/CuriousAFL /home/david/Documents/gits/CuriousAFL/CuriousAFL /home/david/Documents/gits/CuriousAFL/CuriousAFL /home/david/Documents/gits/CuriousAFL/CuriousAFL/CMakeFiles/test-instr0_generator.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test-instr0_generator.dir/depend

