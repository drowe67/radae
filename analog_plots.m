# analog_plots.m
# Octave help script to do a few plots

1;
pkg load statistics;

function do_plots(fn='l.f32',png_fn='')
    l=load_f32(fn,1);
    s=l(1:2:end)+j*l(2:2:end);
    figure(1); clf; plot(s,'.'); axis([-4 4 -4 4])
    if length(png_fn)
        print("-dpng",png_fn);
    end
    figure(2); clf;
    [nn cc] = hist3([real(s) imag(s)],[25 25]);
    mesh(cc{1},cc{2},nn);

endfunction

function multipath_example()
    Nc = 20; Rs = 50; d = 0.002;
    G1 = 1; G2 = 1;
    w = 2*pi*(0:Nc-1);
    H = G1 + G2*exp(-j*w*d*Rs);
    figure(1); clf; plot((0:Nc-1)*Rs, abs(H),'+-');
    title('|H(f)| for test multipath channel');
    xlabel('Freq (Hz)'); ylabel('|H(f)|');
    print("-dpng","multipath_h.png")
endfunction
