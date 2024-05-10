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

    if length(rx_fn)
        rx=load_f32(rx_fn,1); 
        rx=rx(1:2:end)+j*rx(2:2:end); 
        figure(3); clf; plot(rx); title('rx Scatter (IQ)');
        figure(4); clf; plot(abs(rx(1:1000))); xlabel('Time (samples)'); ylabel('|rx|');
        figure(5); clf; plot_specgram(rx, Fs=8000, 0, 3000);
        peak = max(abs(rx).^2);
        av = mean(abs(rx).^2);
        PAPRdB = 10*log10(peak/av);
        printf("Pav: %f PAPRdB: %5.2f\n", av, PAPRdB);
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

