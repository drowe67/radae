message(STATUS "Will build opus with FARGAN")

# The user doesn't have a valid opus, build from source
include(ExternalProject)
ExternalProject_Add(build_opus
    GIT_REPOSITORY https://gitlab.xiph.org/xiph/opus.git
    GIT_TAG main
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND ./autogen.sh && ./configure --enable-dred --disable-shared
    BUILD_COMMAND $(MAKE)
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(build_opus BINARY_DIR)
ExternalProject_Get_Property(build_opus SOURCE_DIR)
add_library(opus SHARED IMPORTED)
add_dependencies(opus build_opus)

set_target_properties(opus PROPERTIES
    IMPORTED_LOCATION "${BINARY_DIR}/.libs/libopus${CMAKE_STATIC_LIBRARY_SUFFIX}"
    IMPORTED_IMPLIB   "${BINARY_DIR}/.libs/libopus${CMAKE_IMPORT_LIBRARY_SUFFIX}"
)

include_directories(${SOURCE_DIR}/dnn ${SOURCE_DIR}/celt ${SOURCE_DIR}/include)
