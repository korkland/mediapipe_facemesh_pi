# FindTFLite.cmake
#
# This module finds a pre-built TensorFlow Lite library and its dependencies.
# It is designed to work with a TFLite library built from source using the
# CMake build process recommended by TensorFlow.
#
# This module defines the following variables:
#
#   TFLite_FOUND          - True if the TFLite library and its dependencies were found.
#   TFLite_INCLUDE_DIRS   - The include directories needed to use TFLite.
#   TFLite_LIBRARIES      - A list of all the libraries to link against. (Legacy)
#
# It also defines the following IMPORTED target:
#
#   TFLite::tflite        - The main imported target. Linking to this target will
#                           automatically handle include directories and all dependent
#                           static libraries.
#
# Hints (User must set these before calling find_package):
# ^^^^^
# A user MUST set the following variables to tell this module where to look:
#
#   TFLITE_BUILD_DIR      - The absolute path to the TFLite build directory
#                           (e.g., /path/to/tensorflow/tflite_build).
#   TENSORFLOW_ROOT       - The absolute path to the TensorFlow source root directory
#                           (e.g., /path/to/tensorflow).

# --- Start of Find Script ---

include(FindPackageHandleStandardArgs)

# 1. Check for user-provided hints
# ---------------------------------
if(NOT TFLITE_BUILD_DIR OR NOT TENSORFLOW_ROOT)
    message(FATAL_ERROR "You must set TFLITE_BUILD_DIR and TENSORFLOW_ROOT before calling find_package(TFLite).")
endif()

# 2. Find the main TFLite library
# -------------------------------
find_library(TFLite_LIBRARY
    NAMES tensorflow-lite
    PATHS ${TFLITE_BUILD_DIR}
    NO_DEFAULT_PATH
)

# 3. Define all include directories
# ---------------------------------
set(TFLite_INCLUDE_DIRS
    ${TENSORFLOW_ROOT}
    ${TFLITE_BUILD_DIR}/flatbuffers/include
)

# 4. Find all dependency libraries
# ---------------------------------
set(_TFLITE_DEPENDENCY_LIBS)

# Use find_library for dependencies that might be in slightly different subdirs
find_library(FLATBUFFERS_LIB flatbuffers PATHS ${TFLITE_BUILD_DIR} PATH_SUFFIXES _deps/flatbuffers-build NO_DEFAULT_PATH)
find_library(XNNPACK_LIB XNNPACK PATHS ${TFLITE_BUILD_DIR} PATH_SUFFIXES _deps/xnnpack-build NO_DEFAULT_PATH)
find_library(PTHREADPOOL_LIB pthreadpool PATHS ${TFLITE_BUILD_DIR} PATH_SUFFIXES pthreadpool NO_DEFAULT_PATH)
find_library(CPUINFO_LIB cpuinfo PATHS ${TFLITE_BUILD_DIR} PATH_SUFFIXES _deps/cpuinfo-build NO_DEFAULT_PATH)

# For libraries with very specific, known paths, we can set them directly.
set(RUY_LIB_DIR ${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy)
set(FFT2D_LIB_DIR ${TFLITE_BUILD_DIR}/_deps/fft2d-build)
set(FARMHASH_LIB ${TFLITE_BUILD_DIR}/_deps/farmhash-build/libfarmhash.a)

# Create a list of all required static libraries in the correct link order.
# The order is important for resolving symbols in static libraries.
set(_TFLITE_DEPENDENCY_LIBS

    # Dependencies found via find_library
    ${FLATBUFFERS_LIB}
    ${XNNPACK_LIB}
    ${PTHREADPOOL_LIB}
    ${CPUINFO_LIB}

    # Core ruy libraries
    ${RUY_LIB_DIR}/libruy_frontend.a
    ${RUY_LIB_DIR}/libruy_context.a
    ${RUY_LIB_DIR}/libruy_context_get_ctx.a
    ${RUY_LIB_DIR}/libruy_ctx.a
    ${RUY_LIB_DIR}/libruy_trmul.a
    ${RUY_LIB_DIR}/libruy_prepare_packed_matrices.a
    ${RUY_LIB_DIR}/libruy_allocator.a
    ${RUY_LIB_DIR}/libruy_thread_pool.a
    ${RUY_LIB_DIR}/libruy_blocking_counter.a
    ${RUY_LIB_DIR}/libruy_wait.a
    ${RUY_LIB_DIR}/libruy_block_map.a
    ${RUY_LIB_DIR}/libruy_system_aligned_alloc.a
    ${RUY_LIB_DIR}/libruy_kernel_arm.a
    ${RUY_LIB_DIR}/libruy_pack_arm.a
    ${RUY_LIB_DIR}/libruy_apply_multiplier.a
    ${RUY_LIB_DIR}/libruy_tune.a
    ${RUY_LIB_DIR}/libruy_cpuinfo.a
    ${RUY_LIB_DIR}/libruy_denormal.a
    ${RUY_LIB_DIR}/libruy_prepacked_cache.a
    ${RUY_LIB_DIR}/libruy_have_built_path_for_avx.a
    ${RUY_LIB_DIR}/libruy_have_built_path_for_avx2_fma.a
    ${RUY_LIB_DIR}/libruy_have_built_path_for_avx512.a
    ${RUY_LIB_DIR}/profiler/libruy_profiler_instrumentation.a
    ${RUY_LIB_DIR}/profiler/libruy_profiler_profiler.a

    # FFT2D libraries
    ${FFT2D_LIB_DIR}/libfft2d_fftsg.a
    ${FFT2D_LIB_DIR}/libfft2d_fftsg2d.a
    ${FFT2D_LIB_DIR}/libfft2d_fftsg3d.a
    ${FFT2D_LIB_DIR}/libfft2d_alloc.a
    ${FFT2D_LIB_DIR}/libfft2d_shrtdct.a
    ${FFT2D_LIB_DIR}/libfft2d_fft4f2d.a

    # FarmHash library
    ${FARMHASH_LIB}

    # System libraries
    pthread
    dl
    m
)

# Set the legacy variables (for compatibility if needed)
set(TFLite_LIBRARIES ${TFLite_LIBRARY} ${_TFLITE_DEPENDENCY_LIBS})

# 5. Handle standard arguments and create the imported target
# -----------------------------------------------------------
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TFLite
    FOUND_VAR TFLite_FOUND
    REQUIRED_VARS TFLite_LIBRARY TFLite_INCLUDE_DIRS
)

if(TFLite_FOUND AND NOT TARGET TFLite::tflite)
    # Create an IMPORTED target. This is the modern CMake way to use libraries.
    add_library(TFLite::tflite UNKNOWN IMPORTED)

    # Set the location of the main .a file
    set_target_properties(TFLite::tflite PROPERTIES
        IMPORTED_LOCATION "${TFLite_LIBRARY}"
    )

    # Set the public include directories for the target
    set_target_properties(TFLite::tflite PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${TFLite_INCLUDE_DIRS}"
    )

    # Set all the dependency libraries. When a user links to TFLite::tflite,
    # the linker will automatically be passed this full list of libraries.
    set_target_properties(TFLite::tflite PROPERTIES
        INTERFACE_LINK_LIBRARIES "${_TFLITE_DEPENDENCY_LIBS}"
    )
endif()

mark_as_advanced(TFLite_LIBRARY TFLite_INCLUDE_DIRS TFLite_LIBRARIES)