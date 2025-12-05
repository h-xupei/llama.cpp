cmake -G "Ninja" -DCMAKE_TOOLCHAIN_FILE="C:
\Users\8356509\AppData\Local\Android\Sdk\ndk\29.0.14206865
\build\cmake\android.toolchain.cmake" -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28
-DCMAKE_C_FLAGS="-march=armv8.7a" -DCMAKE_CXX_FLAGS
="-march=armv8.7a" -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF
-B build-android
cmake --build build-android --config Release
adb shell "su -c 'chmod 777 /data/local/tmp/cpp/llama-runner && cd /data/local/tmp && nohup
./cpp/llama-runner ./cpp/Qwen3-0.6B-Q4_0.gguf > ./cpp/output.log 2>&1 &'"
