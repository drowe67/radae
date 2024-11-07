% Octave script to explore BPF of basband symbols

function bbfm_bpf()
    pkg load signal;
    Fs = 8000; T = 1/Fs; Rs = 2000; M = Fs/Rs; Nsym = 6; alpha = 0.25;
    rn = gen_rn_coeffs(alpha, T, Rs, Nsym, M);
    bpf = fir2(100,[0 250 350 3000 3100 4000]/(Fs/2),[0.001 0.001 1 1 0.001 0.001]);

    figure(1); clf; 
    subplot(211);
    [h,w] = freqz(rn); plot(w*Fs/(2*pi), 20*log10(abs(h))); grid('minor'); ylabel('RRC');
    subplot(212);
    [h,w] = freqz(bpf); plot(w*Fs/(2*pi), 20*log10(abs(h))); grid; ylabel('BPF');

    Nsymb = 1000;
    tx_symb = 1 - 2*(rand(Nsymb,1)>0.5);
    tx_pad = zeros(1,M*Nsymb);
    tx_pad(1:M:end) = tx_symb;
    tx = filter(rn,1,tx_pad);
    tx = filter(bpf,1,tx)
    rx = filter(rn,1,tx);
    rx_symb = rx(1:M:end);
    figure(2); clf; 
    subplot(211); stem(tx_symb(1:100)); ylabel('Tx symbols');
    subplot(212); stem(rx_symb(1:100)); ylabel('Rx Symbols');

    figure(3); clf;
    plot(20*log10(abs(fft(tx)(1:length(tx)/2))))
end

