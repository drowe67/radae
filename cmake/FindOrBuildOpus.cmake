message(STATUS "Testing opus at ${OPUS_DIR} for FARGAN")

if(OPUS_DIR)
    # This ensures that the version of opus found on the user's system
    # has FARGAN enabled (--enable-dred when building opus). If our test
    # file can't be built, we can't use this version of opus.
    try_compile(OPUS_HAS_FARGAN
        SOURCES ${CMAKE_SOURCE_DIR}/cmake/test_opus.c
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${OPUS_DIR}/dnn;${OPUS_DIR}/celt;${OPUS_DIR}/include"
        LINK_LIBRARIES ${OPUS_DIR}/.libs/libopus.a)
endif(OPUS_DIR)

if(NOT OPUS_HAS_FARGAN)
    message(STATUS "  Could not find usable opus with FARGAN, will build it ourselves")

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
else(NOT OPUS_HAS_FARGAN)
    message(STATUS "  Using opus at ${OPUS_DIR}")
    include_directories(${OPUS_DIR}/dnn ${OPUS_DIR}/celt ${OPUS_DIR}/include)

    add_library(opus SHARED IMPORTED)
    set_target_properties(opus PROPERTIES
        IMPORTED_LOCATION "${OPUS_DIR}/.libs/libopus${CMAKE_STATIC_LIBRARY_SUFFIX}"
        IMPORTED_IMPLIB   "${OPUS_DIR}/.libs/libopus${CMAKE_IMPORT_LIBRARY_SUFFIX}"
    )
endif(NOT OPUS_HAS_FARGAN)
