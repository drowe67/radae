# radae_plots.m
# Octave helper script to generate some plots form inference.py outputs

1;
pkg load statistics signal;

function do_plots(z_fn='l.f32',rx_fn='', png_fn='')
    z=load_f32(z_fn,1);
    s=z(1:2:end)+j*z(2:2:end);
    figure(1); clf; plot(s,'.'); title('Scatter');
    mx = max(abs(z))*1.5; axis([-mx mx -mx mx])
    if length(png_fn)
        print("-dpng",sprintf("%s_scatter.png",png_fn));
    end
    figure(2); clf;
    [nn cc] = hist3([real(s) imag(s)],[25 25]);
    mesh(cc{1},cc{2},nn); title('Scatter 3D');
    if length(png_fn)
        print("-dpng",sprintf("%s_scatter_3d.png",png_fn));
    end
    figure(3); clf; hist(abs(s));
 
    if length(rx_fn)
        rx=load_f32(rx_fn,1); 
        rx=rx(1:2:end)+j*rx(2:2:end); 
        figure(4); clf; plot(rx); title('rate Fs Scatter (IQ)'); mx = max(abs(rx))*1.5; axis([-mx mx -mx mx]);
        figure(5); clf; plot(abs(rx(1:1000))); xlabel('Time (samples)'); ylabel('|rx|');
        figure(6); clf; plot_specgram(rx, Fs=8000, 0, 3000);
        peak = max(abs(rx).^2);
        av = mean(abs(rx).^2);
        PAPRdB = 10*log10(peak/av);
        peak = max(abs(rx(1:160)).^2);
        av = mean(abs(rx(1:160)).^2);
        PilotPAPRdB = 10*log10(peak/av);
        printf("Pav: %f PAPRdB: %5.2f PilotPAPRdB: %5.2f\n", av, PAPRdB, PilotPAPRdB);
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

% Plots loss v Eb/No curves from text files dumped by train.py, pass in pairs of EqNo_file.txt,legend
function loss_EqNo_plot(png_fn, varargin)
    figure(1); clf; hold on;
    i = 1;
    while i <= length(varargin)
        fn = varargin{i};
        data = load(fn);
        i++; leg = varargin{i}; leg = strrep (leg, "_", " ")
        plot(data(:,1),data(:,2),sprintf("+-;%s;",leg))
        i++;
    end
    hold off; grid; xlabel('Eq/No (dB)'); ylabel('loss');
    if length(png_fn)
        print("-dpng",png_fn);
    end
endfunction

% Plots loss v C/No curves from text files dumped by train.py, pass in EqNo_file.txt,dim,leg for each curve
function loss_CNo_plot(png_fn, Rs, B, varargin)
    figure(1); clf; hold on;
    i = 1;
    while i <= length(varargin)
        fn = varargin{i};
        data = load(fn);
        i++; dim = varargin{i}; Nc = dim/2;
        i++; leg = varargin{i}; leg = strrep (leg, "_", " ");
        EqNo = data(:,1);
        CNo = EqNo + 10*log10(Rs*Nc/B);
        plot(CNo,data(:,2),sprintf("+-;%s;",leg))
        i++;
    end
    hold off; grid; 
    if B==1
      xlabel('C/No (dB)');
    else
      xlabel('SNR (dB)');
    end
     ylabel('loss');
    if length(png_fn)
        print("-dpng",png_fn);
    end
endfunction

% usage:
%   radae_plots; ofdm_sync_plots("","ofdm_sync.txt","go-;genie;","ofdm_sync_pilot_eq.txt","r+-;mean6;","ofdm_sync_pilot_eq_f2.txt","bx-;mean6 2 Hz;","ofdm_sync_pilot_eq_g0.1.txt","gx-;mean6 gain 0.1;","ofdm_sync_pilot_eq_ls.txt","ro-;LS;","ofdm_sync_pilot_eq_ls_f2.txt","bo-;LS 2 Hz;")

function ofdm_sync_plots(epslatex, varargin)
    if length(epslatex)
        [textfontsize linewidth] = set_fonts();
    end
    figure(1); clf; hold on;
    EbNodB = -8:4; EbNo = 10.^(EbNodB/10);
    awgn_theory = 0.5*erfc(sqrt(EbNo));
    multipath_theory = 0.5.*(1-sqrt(EbNo./(EbNo+1)));
    plot(EbNodB, awgn_theory,'b+-;AWGN theory;');
    plot(EbNodB, multipath_theory,'bx-;Multipath theory;');
    i = 1;
    while i <= length(varargin)
        fn = varargin{i};
        data = load(fn);
        i++; leg = varargin{i}; leg = strrep (leg, "_", " ")
        plot(data(:,1),data(:,2),sprintf("%s",leg))
        i++;
    end
    hold off; grid('minor'); xlabel('Eb/No (dB)'); ylabel('BER'); legend('boxoff');
    if length(epslatex)
        print_eps_restore(epslatex,"-S350,300",textfontsize,linewidth);
    end
endfunction

function [textfontsize linewidth] = set_fonts(font_size=12)
  textfontsize = get(0,"defaulttextfontsize");
  linewidth = get(0,"defaultlinelinewidth");
  set(0, "defaulttextfontsize", font_size);
  set(0, "defaultaxesfontsize", font_size);
  set(0, "defaultlinelinewidth", 0.5);
end

function restore_fonts(textfontsize,linewidth)
  set(0, "defaulttextfontsize", textfontsize);
  set(0, "defaultaxesfontsize", textfontsize);
  set(0, "defaultlinelinewidth", linewidth);
end

function print_eps_restore(fn,sz,textfontsize,linewidth)
  print(fn,sz,"-depslatex");
  printf("printing... %s\n", fn);
  restore_fonts(textfontsize,linewidth);
end

% test expressions for sigma calculation with rate Fs bottleneck
function test_rate_Fs_bottleneck
  Nc=10; Fs=8000; Rs=50; M=Fs/Rs;
  B = 1/sqrt(Nc);
  x = zeros(1,M);
  n = 0:M-1;
  for c=1:Nc
    x += B*exp(j*2*pi*n*c/M);
  end

  % check average rate Fs power == 1. Power is energy per unit time. Note this
  % test waveform will have a high peak power, as no attempt has been made to
  % manage PAPR
  Px = sum(abs(x).^2)/M;
  printf("Power of time domain signal Px: %f (target: 1.0)\n",Px);
  
  % check symbol amplitude for one carrier
  c = 1;
  Aq = abs(sum(x .* exp(-j*2*pi*n*c/M)));
  printf("Amplitude of carrier %d freq domain PSK symbol Aq: %f (target %f)\n",c, Aq,B*M);

  % check symbol SNR
  EbNo_target = 1;
  sigma = M/sqrt(2*Nc*EbNo_target);
  EqNo = (Aq^2)/(sigma^2);
  EbNo = 0.5*EqNo;
  printf("EbNo: %f (target: %f)\n",EbNo,EbNo_target);
  
end

% Latex plotting for SNR estimator. run est_snr.py first
function est_snr_plot(epslatex="")
    if length(epslatex)
        [textfontsize linewidth] = set_fonts();
    end
    points = load("est_snr.txt");

    % point for mean line
    mean_x = [];
    mean_y = [];
    for snrdB=-10:20
      x = find(points(:,1) == snrdB);
      mean_x = [mean_x; snrdB];
      mean_y = [mean_y; mean(points(x,2))];
    end

    figure(1); clf;
    plot(points(:,1), points(:,2),'b.');
    hold on;
    plot(mean_x,mean_y,'ro-');
    hold off;
    hold off; grid('minor'); xlabel('SNR (dB)'); ylabel('SNR Est (dB)');
    if length(epslatex)
        print_eps_restore(epslatex,"-S300,250",textfontsize,linewidth);
    end
endfunction
