# generate time domain multipath simulation files if they don't exist
if [ ! -f g_mpg.f32 ]; then
  DISPLAY="" echo "Fs=8000; Rs=50; Nc=20; multipath_samples('mpg', Fs, Rs, Nc, 120, '','g_mpg.f32'); quit" | octave-cli -qf
fi
if [ ! -f g_mpp.f32 ]; then
  DISPLAY="" echo "Fs=8000; Rs=50; Nc=20; multipath_samples('mpp', Fs, Rs, Nc, 120, '','g_mpp.f32'); quit" | octave-cli -qf
fi
if [ ! -f g_mpd.f32 ]; then
  DISPLAY="" echo "Fs=8000; Rs=50; Nc=20; multipath_samples('mpd', Fs, Rs, Nc, 120, '','g_mpd.f32'); quit" | octave-cli -qf
fi
if [ ! -f g_mpp_low.f32 ]; then
  DISPLAY="" echo "multipath_samples('mpp_low', 8000, 50, 14, 10, '', 'g_mpp_low.f32'); quit" | octave-cli -qf
fi
if [ ! -f g_mpp_high.f32 ]; then
  DISPLAY="" echo "multipath_samples('mpp_high', 8000, 50, 14, 10, '', 'g_mpp_high.f32'); quit" | octave-cli -qf
fi
