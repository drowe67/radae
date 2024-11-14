include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(lpcnet_demo lpcnet_demo.c)
target_link_libraries(lpcnet_demo opus m)

add_library(rade rade_api.c)
target_link_libraries(rade Python3::Python)
target_include_directories(rade PRIVATE
    "$<TARGET_PROPERTY:Python3::NumPy,INTERFACE_INCLUDE_DIRECTORIES>")
target_compile_definitions(rade PRIVATE -DIS_BUILDING_RADE_API=1)
set_target_properties(rade PROPERTIES
    SOVERSION "0.1"
    PUBLIC_HEADER "rade_api.h"
)

add_executable(radae_tx radae_tx.c)
target_link_libraries(radae_tx rade Python3::Python)
target_include_directories(radae_tx PRIVATE
    "$<TARGET_PROPERTY:Python3::NumPy,INTERFACE_INCLUDE_DIRECTORIES>")

add_executable(radae_rx radae_rx.c)
target_link_libraries(radae_rx rade Python3::Python)
target_include_directories(radae_rx PRIVATE
    "$<TARGET_PROPERTY:Python3::NumPy,INTERFACE_INCLUDE_DIRECTORIES>")

add_library(radecore rade_enc.c rade_dec.c rade_enc_data.c rade_dec_data.c)
set_target_properties(radecore PROPERTIES
    SOVERSION "0.1"
    PUBLIC_HEADER "rade_api.h"
)

add_executable(test_rade_enc test_rade_enc.c)
target_link_libraries(test_rade_enc radecore opus m)

add_executable(test_rade_dec test_rade_dec.c)
target_link_libraries(test_rade_dec radecore opus m)

add_executable(write_rade_weights write_rade_weights.c)
target_link_libraries(write_rade_weights radecore opus m)
