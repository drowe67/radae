include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(lpcnet_demo lpcnet_demo.c)
target_link_libraries(lpcnet_demo opus m)

add_library(rade rade_api.c)
target_link_libraries(rade Python3::Python Python3::NumPy)
set_target_properties(rade PROPERTIES
    SOVERSION "0.1"
    PUBLIC_HEADER "rade_api.h"
)

add_executable(radae_tx radae_tx.c rade_api.c)
target_link_libraries(radae_tx Python3::Python Python3::NumPy)

add_executable(radae_rx radae_rx.c rade_api.c)
target_link_libraries(radae_rx Python3::Python Python3::NumPy)

