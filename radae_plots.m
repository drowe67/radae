# radae_plots.m
# Octave helper script to generate some plots form inference.py outputs

1;
pkg load statistics signal;

function do_plots(z_fn='l.f32',rx_fn='', png_fn='')
    z=load_f32(z_fn,1);
    s=z(1:2:end)+j*z(2:2:end);
    figure(1); clf; plot(s,'.'); title('Scatter');
    mx = max(abs(z)); axis([-mx mx -mx mx])
    #if length(png_fn)
    #    print("-dpng",png_fn);
    #end
    figure(2); clf;  title('Scatter');
    [nn cc] = hist3([real(s) imag(s)],[25 25]);
    mesh(cc{1},cc{2},nn);
    title('Scatter');

    if length(rx_fn)
        rx=load_f32('rx.f32',1); 
        rx=rx(1:2:end)+j*rx(2:2:end); 
        figure(3); clf; plot(rx); title('rx Scatter (IQ)');
        figure(4); clf; plot(abs(rx(1:1000))); xlabel('Time (samples)'); ylabel('|rx|');
        figure(5); clf; plot_specgram(rx, Fs=8000, 0, 2000);
    end
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
