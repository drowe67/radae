% multipath_samples.m
% create rate Rs H and rate Fs G matrices for multipath channel simulation,
% saved as flattened .f32 files

function multipath_samples(ch, Fs, Rs, Nc, Nseconds, H_fn, G_fn="")
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
        printf("Unknown channel type!")
        return
    end
        
    G1 = doppler_spread(dopplerSpreadHz, Fs, nsam).';
    G2 = doppler_spread(dopplerSpreadHz, Fs, nsam).';
    
    % approximation to normalise power through HF channel
    hf_gain = 1.0/sqrt(var(G1)+var(G2));

    % H matrix of magnitude samples, timesteps along rows, carrier alongs cols
    % sampled at rate Rs (ons sample per symbol)

    M = Fs/Rs;
    omega = 2*pi*(0:Nc-1);
    d = path_delay_s;
    H = hf_gain*abs(G1(1:M:end) + G2(1:M:end).*exp(-j*omega*d*Rs));
    figure(1); mesh(H(1:10*Rs,:))
    f=fopen(H_fn,"wb");
    [r c] = size(H);
    Hflat = reshape(H', 1, r*c);
    fwrite(f, Hflat, 'float32');
    fclose(f);

    if length(G_fn)
        % G matrix cols are G1 G2, rows timesteps, with hf_gain the first row,
        % stored as flat ...G1G2G1G2... complex samples 

        len_samples = length(G1);
        G = zeros(1,len_samples*4);
        G(1:4) = hf_gain;
        for i=1:len_samples
            G(i*4+1) = real(G1(i));
            G(i*4+2) = imag(G1(i));
            G(i*4+3) = real(G2(i));
            G(i*4+4) = imag(G2(i));
        end
        f = fopen(G_fn,"wb");
        fwrite(f, G, "float32");
        fclose(f);
    end
endfunction
