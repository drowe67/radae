if [ ! -f g_mpg.f32 ]; then
  DISPLAY="" echo "Fs=8000; Rs=50; Nc=20; multipath_samples('mpg', Fs, Rs, Nc, 120, 'h_nc20_mpp.f32','g_mpg.f32'); quit" | octave-cli -qf
fi
if [ ! -f g_mpp.f32 ]; then
  DISPLAY="" echo "Fs=8000; Rs=50; Nc=20; multipath_samples('mpp', Fs, Rs, Nc, 120, 'h_nc20_mpp.f32','g_mpp.f32'); quit" | octave-cli -qf
fi
