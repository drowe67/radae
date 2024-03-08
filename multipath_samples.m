% multipath_samples.m
% create a .f32 file of multipath channel magnitude samples

function multipath_samples(ch, Fs, Rs, Nc, Nseconds, fn)
    pkg load signal;
    nsam = Fs*Nseconds;
    randn('seed',1);

    printf("Generating HF model spreading samples...\n")
    if strcmp(ch,"mpg")
        dopplerSpreadHz = 0.1; path_delay_s = 0.5E-3;
    elseif strcmp(ch,"mpp")
        dopplerSpreadHz = 1.0; path_delay_s = 2E-3;
    elseif strcmp(ch,"mpd")
        dopplerSpreadHz = 2.0; path_delay_s = 4E-3;
    else
        printf("Uknown channel type!")
        return
    end
        
    G1 = doppler_spread(dopplerSpreadHz, Fs, nsam).';
    G2 = doppler_spread(dopplerSpreadHz, Fs, nsam).';
    
    % approximation to normalise power through HF channel
    hf_gain = 1.0/sqrt(var(G1)+var(G2));

    omega = 2*pi*(0:Nc-1)*Rs/Fs;
    d = path_delay_s;
    H = hf_gain*abs(G1 + G2.*exp(-j*omega*d*Rs));
    size(H)
    figure(1); mesh(H)
    f=fopen(fn,"wb");
    [r c] = size(H);
    Hflat = reshape(H', 1, r*c);
    fwrite(f, Hflat, 'float32');
    fclose(f);
endfunction
