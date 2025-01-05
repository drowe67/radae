% multipath_samples.m
% create rate Rs H and rate Fs G matrices for multipath channel simulation,
% saved as flattened .f32 files

function multipath_samples(ch, Fs, Rs, Nc, Nseconds, H_fn, G_fn="",H_complex=0)
    pkg load signal;
    nsam = Fs*Nseconds;
    randn('seed',1);

    printf("Generating Doppler spreading samples...\n")
    if strcmp(ch,"mpg")
        dopplerSpreadHz = 0.1; path_delay_s = 0.5E-3;
    elseif strcmp(ch,"mpp")
        dopplerSpreadHz = 1.0; path_delay_s = 2E-3;
    elseif strcmp(ch,"mpd")
        dopplerSpreadHz = 2.0; path_delay_s = 4E-3;
    elseif strcmp(ch,"lmr60")
        % 60 km/hr, 450 MHz
        fd = 450E6*(60*1E3/3600/3E8)
        dopplerSpreadHz = 2*fd;
        path_delay_s = 200E-6
    else
        printf("Unknown channel type!")
        return
    end
        
    G1 = doppler_spread(dopplerSpreadHz, Fs, nsam).';
    G2 = doppler_spread(dopplerSpreadHz, Fs, nsam).';
    
    % approximation to normalise power through HF channel
    hf_gain = 1.0/sqrt(var(G1)+var(G2));

    % H matrix of magnitude samples, timesteps along rows, carrier alongs cols
    % sampled at rate Rs (one sample per symbol).

    M = Fs/Rs;
    omega = 2*pi*(0:Nc-1);
    d = path_delay_s;
    H = hf_gain*(G1(1:M:end) + G2(1:M:end).*exp(-j*omega*d*Rs));
    figure(1); clf;
    if Nc > 1 
      mesh(abs(H(1:10*Rs,:)))
    else
      # single carrier case
      H = abs(H);
      Nsecplot=1
      subplot(211); plot(H(1:Nsecplot*Rs,:)); xlabel('Symbols'); ylabel('|H|')
      subplot(212); plot(20*log10(H(1:Nsecplot*Rs,:))); xlabel('Symbols'); ylabel('|H| (dB)')
      Pav = mean(H.^2)
      P = sqrt(1.0)
      LCR_theory = ((2*pi*P/Pav)^0.5)*fd*exp(-P/Pav)
      LC = 0;
      LC_log = [];
      for n=1:length(H)-1
        p1 = H(n,1).^2;
        p2 = H(n+1,1).^2;
        if p1 < P && p2 > P
          LC++;
          LC_log = [LC_log n];
        end
      end
      LCR_meas = LC/Nseconds
      subplot(211); hold on; stem(LC_log,sqrt(P)*ones(length(LC_log))); hold off; axis([0 Nsecplot*Rs 0 3]);
    end
    if H_complex
      bytes_per_sample = 8
    else
      bytes_per_sample = 4
    end
    printf("H file size is Nseconds*Rs*Nc*(%d bytes/sample) = %d*%d*%d*%d = %d bytes\n", bytes_per_sample,
           Nseconds,Rs,Nc,bytes_per_sample, Nseconds*Rs*Nc*bytes_per_sample)
    f=fopen(H_fn,"wb");
    [r c] = size(H);
    Hflat = reshape(H', 1, r*c);
    if H_complex
      tmp = zeros(2*length(Hflat),1);
      tmp(1:2:end) = real(Hflat);
      tmp(2:2:end) = imag(Hflat);
      Hflat = tmp;
    else
      Hflat = abs(Hflat);
    end
    fwrite(f, Hflat, 'float32');
    fclose(f);

    if length(G_fn)
        % G matrix cols are G1 G2, rows timesteps, with hf_gain the first row,
        % stored as flat ...G1G2G1G2... complex samples 

        len_samples = length(G1);
        G = zeros(1,(1+len_samples)*4);
        G(1:4) = hf_gain;
        G(5:4:end) = real(G1);
        G(6:4:end) = imag(G1);
        G(7:4:end) = real(G2);
        G(8:4:end) = imag(G2);
        printf("G file size is (Nseconds*Fs+1)*(2 complex samples)*(8 bytes/sample) = %d*2*8 = %d bytes\n", (Nseconds*Fs+1),(Nseconds*Fs+1)*2*8)
        f = fopen(G_fn,"wb");
        fwrite(f, G, "float32");
        fclose(f);
    end
endfunction
