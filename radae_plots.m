# radae_plots.m
# Octave helper script to generate some plots form inference.py outputs

1;
pkg load statistics signal;

function do_plots(z_fn='l.f32',rx_fn='', png_fn='', epslatex='')
    if length(epslatex)
        [textfontsize linewidth] = set_fonts(20);
    end
    if length(z_fn)
      z=load_f32(z_fn,1);
      s=z(1:2:end)+j*z(2:2:end);
      figure(2); clf; plot(s,'.'); title('Scatter');
      mx = max(abs(z))*1.2; axis([-mx mx -mx mx])
      if length(png_fn)
        print("-dpng",sprintf("%s_scatter.png",png_fn));
      end
      if length(epslatex)
        print_eps(sprintf("%s_scatter.eps",epslatex),"-S250,250");
      end
      figure(3); clf;
      [nn cc] = hist3([real(s) imag(s)],[25 25]);
      mesh(cc{1},cc{2},nn); title('Scatter 3D');   
      if length(png_fn)
        print("-dpng",sprintf("%s_scatter_3d.png",png_fn));
      end
      if length(epslatex)
        print_eps(sprintf("%s_scatter_3d.eps",epslatex),"-S300,300");
      end
      figure(4); clf; hist(abs(s));
    end
    
    if length(rx_fn)
        rx=load_f32(rx_fn,1); 
        rx=rx(1:2:end)+j*rx(2:2:end); 
        
        tx_bpf = 0;
        if tx_bpf
          lpf=fir1(100,900/4000);
          w = 2*pi*1500/8000;
          N=length(rx);
          lo = exp(j*(0:N-1)*w)';
          rx = filter(lpf,1,rx.*lo).*conj(lo);
          ind = find(abs(rx) > 1);
          rx(ind) = exp(j*angle(rx(ind)));
        end
        figure(5); clf; plot(rx); title('rate Fs Scatter (IQ)'); mx = max(abs(rx))*1.5; axis([-mx mx -mx mx]);
        figure(6); clf; plot(real(rx)); xlabel('Time (samples)'); ylabel('rx');
        figure(7); clf; plot_specgram(rx, Fs=8000, 0, 3000);
        
        max(abs(rx).^2)
        mean(abs(rx).^2)
        peak = max(abs(rx).^2);
        av = mean(abs(rx).^2);
        PAPRdB = 10*log10(peak/av);
        peak = max(abs(rx(1:160)).^2);
        av = mean(abs(rx(1:160)).^2);
        PilotPAPRdB = 10*log10(peak/av);
        printf("Pav: %f PAPRdB: %5.2f PilotPAPRdB: %5.2f\n", av, PAPRdB, PilotPAPRdB);
 
        fcentre = 1475;
        bwHz = bandwidth(rx, fcentre)

        % Spectrum plot
        Fs = 8000; y = pwelch(rx,[],[],1024,Fs); y_dB = 10*log10(y);
        mx = max(y_dB); mx = ceil(mx/10)*10
        figure(8); clf; 
        plot((0:length(y)-1)*Fs/length(y),y_dB-mx);
        hold on;
        plot([fcentre-bwHz/2 fcentre-bwHz/2 fcentre+bwHz/2 fcentre+bwHz/2 fcentre-bwHz/2 ],[-35 -5 -5 -35 -35],'r-');
        hold off;
        axis([0 3000 -40 0]); grid; xlabel('Freq (Hz)'); ylabel('dB');
        if length(epslatex)
          print_eps_restore(sprintf("%s_psd.eps",epslatex),"-S300,200",textfontsize,linewidth);
        end

    end
endfunction

function do_plots_bbfm(z1_fn, z2_fn="", png_fn='')
    z1=load_f32(z1_fn,1);
    figure(1); clf; 
    stem(z1(1:40),'g');
    if length(z2_fn) 
      z2=load_f32(z2_fn,1);
      hold on;
      stem(z2(1:40),'r');
      hold off;
    end
    title('Rx Symbols');
    if length(png_fn)
      print("-dpng",sprintf("%s.png",png_fn));
    end
endfunction


function p = spec_power(y, centre, bandwidth)
  n = length(y);
  st = round(centre - bandwidth/2); st = max(1,st);
  en = round(centre + bandwidth/2); 
  p = sum(y(st:en));
endfunction


function bwHz = bandwidth(rx, fcentre)
  Nfft = 1204;
  Fs = 8000; y = pwelch(rx,[],[],Nfft,Fs); y_dB = 10*log10(y);
  figure(1);
  plot((0:length(y)-1)*Fs/Nfft,y_dB);

  % 99% power bandwidth
  total_power = sum(y);
  centre = round(Nfft*fcentre/Fs);
  bw = 1;
  do
    bw++;
    p = spec_power(y, centre, bw);
  until p > 0.99*total_power
  bwHz =  bw*Fs/Nfft;
  printf("bandwidth (Hz): %f power/total_power: %f\n", bwHz, p/total_power);

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
function loss_EqNo_plot(png_fn, epslatex, varargin)
    if length(epslatex)
        [textfontsize linewidth] = set_fonts();
    end
    figure(1); clf; hold on;
    i = 1;
    while i <= length(varargin)
        fn = varargin{i};
        data = load(fn);
        i++; leg = varargin{i}; leg = strrep (leg, "_", " ")
        plot(data(:,1),data(:,2),sprintf("+-;%s;",leg))
        i++;
    end
    hold off; grid('minor'); xlabel('Eq/No (dB)'); ylabel('loss'); legend('boxoff');
    mn = min(data(:,1))
    axis([mn mn+25 0.05 0.25])
    if length(png_fn)
        print("-dpng",png_fn);
    end
    if length(epslatex)
        print_eps_restore(epslatex,"-S350,300",textfontsize,linewidth);
    end
endfunction

% Plots loss v C/No curves from text files dumped by train.py, pass in EqNo_file.txt,dim,leg for each curve
function loss_CNo_plot(png_fn, epslatex, Rs, B, varargin)
    if length(epslatex)
        [textfontsize linewidth] = set_fonts(20);
    end
    figure(1); clf; hold on;
    i = 1;
    mn = 100;
    while i <= length(varargin)
        fn = varargin{i};
        data = load(fn);
        i++; dim = varargin{i}; Nc = dim/2;
        i++; leg = varargin{i}; leg = strrep (leg, "_", " ");
        EqNo = data(:,1);
        CNo = EqNo + 10*log10(Rs*Nc/B);
        mn = min([mn; CNo]);
        plot(CNo,data(:,2),sprintf("+-;%s;",leg))
        i++;
    end
    hold off; grid('minor'); 
    if B==1
      xlabel('C/No (dB)');
    else
      xlabel('SNR (dB)');
    end
    ylabel('loss');
    mn = floor(mn);
    axis([mn mn+25 0.05 0.25])
    if length(png_fn)
        print("-dpng",png_fn);
    end
    if length(epslatex)
        print_eps_restore(epslatex,"-S300,200",textfontsize,linewidth);
    end
endfunction

% Plots loss v SNR3k curves from text files dumped by inference.py, see compare_models_inf.py
% pnsr flag optionally includes PAPR
function loss_SNR3k_plot(pnsr=0,png_fn, epslatex, varargin)
    if length(epslatex)
        [textfontsize linewidth] = set_fonts(20);
    end
    figure(1); clf; hold on;
    i = 1;
    mn = 100;
    while i <= length(varargin)
        fn = varargin{i};
        data = load(fn);
        i++; leg = varargin{i}; leg = strrep (leg, "_", " ");
        SNR3k = data(:,1);
        if pnsr
          SNR3k += data(:,3);
        end
        mn = min([mn; SNR3k]);
        plot(SNR3k,data(:,2),sprintf("+-;%s;",leg))
        i++;
    end
    hold off; grid('minor');
    if pnsr
      xlabel('PNR (dB)');
    else
      xlabel('SNR (dB)');
    end
    ylabel('loss');
    mn = floor(mn);
    axis([-5 20 0.05 0.35])
    if length(png_fn)
        print("-dpng",png_fn);
    end
    if length(epslatex)
        print_eps_restore(epslatex,"-S300,200",textfontsize,linewidth);
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

function print_eps(fn,sz)
  print(fn,sz,"-depslatex");
  printf("printing... %s\n", fn);
end

function print_eps_restore(fn,sz,textfontsize,linewidth)
  print_eps(fn,sz);
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

% Latex plotting for SNR estimator. run est_snr_curves.sh first
function est_snr_plot(epslatex="")
    if length(epslatex)
        [textfontsize linewidth] = set_fonts();
    end
    awgn = load("est_snr_awgn.txt");
    mpg = load("est_snr_mpg.txt");
    mpp = load("est_snr_mpp.txt");
    
    figure(1); clf;
    plot(awgn(:,1), awgn(:,1),'bk-');
    hold on;
    plot(awgn(:,1), awgn(:,2),'g.');
    [m b] = linreg(awgn(:,1), awgn(:,2),length(awgn(:,1)));
    plot(awgn(:,1), m*awgn(:,1)+b,'b-'); hold off;
    grid('minor'); xlabel('EsNo (dB)'); ylabel('EsNo Est (dB)');
    axis([-5 20 -5 20]);
    if length(epslatex)
        print_eps(sprintf("%sa",epslatex),"-S300,250",textfontsize,linewidth);
    end
   
    figure(2); clf;
    plot(awgn(:,1), awgn(:,1),'bk--;Ideal;');
    hold on;
    [m b] = linreg(awgn(:,1), awgn(:,2),length(awgn(:,1)));
    plot(awgn(:,1), m*awgn(:,1)+b,'g-;AWGN;');
    [m b] = linreg(mpg(:,1), mpg(:,2),length(mpg(:,1)));
    plot(mpg(:,1),m*mpg(:,1)+b,'b-;MPG;');
    [m b] = linreg(mpp(:,1), mpp(:,2),length(mpp(:,1)));
    plot(mpp(:,1),m*mpp(:,1)+b,'r-;MPP;');
    hold off;
    axis([-5 20 -5 20]); legend('location','southeast'); legend('boxoff');
    grid('minor'); xlabel('EsNo (dB)'); ylabel('EsNo Est (dB)');
    if length(epslatex)
        print_eps_restore(sprintf("%sb",epslatex),"-S300,250",textfontsize,linewidth);
    end
endfunction

% plot D(t,f) surface from rx.py
function D_plot(fn, r=40, c=960)
    D=load_f32(fn,1);
    D=D(1:2:end)+j*D(2:2:end);
    frames = length(D)/(r*c)
    f = 1
    k = ' ';
    do
        figure(1);
        st = (f-1)*(r*c) + 1; en = st + r*c - 1;
        Df=reshape(D(st:en),r,c);
        mesh(abs(Df))
        printf("\rframe: %d  menu: n-next  b-back  q-quit", f);
        fflush(stdout);
        k = kbhit();
        if k == 'n'
            if f < frames; f = f + 1; endif
        endif
        if k == 'b';
            if f > 0; f = f - 1; endif
        endif
    until (k == 'q')
    printf("\n");
end

function p = rayleigh_pdf(sigma_r,x)
  p = (x./(sigma_r*sigma_r)).*exp(-(x.^2)/(2*sigma_r*sigma_r));
end

% checking our scale parameter mapping for Rayleigh
function test_rayleigh(epslatex="")
  randn('seed',1);
  N = 10E6;
  sigma_n = 1;
  noise1 = (sigma_n/sqrt(2))*(randn(1,N) + j*randn(1,N));
  noise2 = (sigma_n/sqrt(2))*(randn(1,N) + j*randn(1,N));
  X1 = abs(noise1);
  X12 = abs(noise1) + abs(noise2);
  
  [h1 x1] = hist(X1,50);
  [h12 x12] = hist(X12,50);
  
  % PDF -------------------

  % est scale param from mean of X1
  sigma1_r = mean(X1)/sqrt(pi/2);
  sigma12_r = sqrt(2)*sigma1_r;
  p1 = rayleigh_pdf(sigma1_r, x1);
  p12 = rayleigh_pdf(sigma12_r, x12);

  if length(epslatex)
    [textfontsize linewidth] = set_fonts();
  end

  warning ("off", "Octave:negative-data-log-axis");
  
  figure(1); clf;
  semilogy(x1,h1/trapz(x1,h1),'b;histogram X1;');
  hold on;
  semilogy(x1,p1,'b+; X1 PDF;');
  semilogy(x12,h12/trapz(x12,h12),'g;histogram X1+X2;');
  semilogy(x12,abs(p12),'g+; X1+X2 PDF;');
  hold off; grid; axis([0 ceil(max(x12)) 1E-6 1]); legend('boxoff');
  xlabel('x'); ylabel('$f(x)$');
  if length(epslatex)
    print_eps_restore(sprintf("%s_pdf", epslatex),"-S300,250",textfontsize,linewidth);
  end

  % P(X1>x) = 1 - CDF(x)

  P1 = exp(-(x1.^2/(2*sigma1_r^2)));
  for i = 1:length(x1)
    P1hist(i) = length(find(X1 > x1(i)))/N;
  end
  
  P12 = 5*exp(-(x12.^2/(4*sigma1_r^2)));
  for i = 1:length(x1)
    P12hist(i) = length(find(X12 > x12(i)))/N;
  end
  
  figure(2); clf;
  semilogy(x1, P1hist, "b;Histogram $P(X1>x)$;");
  hold on;
  semilogy(x1, P1, "b+;$P(X1>x)$;");
  semilogy(x12, P12hist, "g;Histogram $P(X1+X2>x)$;");
  semilogy(x12, P12, "g+;$P(X1+X2>x)$;");
  hold off;
  grid; axis([0 ceil(max(x12)) 1E-6 1]); legend('boxoff');
  xlabel('x'); ylabel('$P(RV>x)$');
  if length(epslatex)
    print_eps_restore(epslatex,"-S300,250",textfontsize,linewidth);
  end
end

function y = relu(x)
  y = x;
  y(find(x<0)) = 0;
end

% Plot SNR v CNR for FM demod model
function plot_SNR_CNR(epslatex="")
    if length(epslatex)
        [textfontsize linewidth] = set_fonts();
    end
    figure(1); clf; hold on;
    fd=5000; fm=3000; 
    beta= fd/fm;
    Gfm=10*log10(3*(beta^2)*(beta+1))
    BWfm = 2*(fd+fm);

    % vanilla implementation of curve
    CNRdB=0:20;
    for i=1:length(CNRdB)
      if CNRdB(i) >= 12
        SNRdB(i) = CNRdB(i) + Gfm;
      else
        SNRdB(i) = (1+Gfm/3)*CNRdB(i) - 3*Gfm;
      end
    end

    % implementation using relus (suitable for PyTorch)
    SNRdB_relu = relu(CNRdB-12) + 12 + Gfm;
    SNRdB_relu += -relu(-(CNRdB-12))*(1+Gfm/3);

    plot(CNRdB,SNRdB,'g;FM;'); 
    plot(CNRdB,SNRdB_relu,'r+;FM relu;'); 
    SSBdB = CNRdB + 10*log10(BWfm) - 10*log10(fm);
    plot(CNRdB,SSBdB,'b;SSB;'); 
    axis([min(CNRdB) max(CNRdB) 10 30]);
    hold off; grid('minor'); xlabel('CNR (dB)'); ylabel('SNR (dB)'); legend('boxoff'); legend('location','northwest');
    if length(epslatex)
        print_eps_restore(epslatex,"-S300,300",textfontsize,linewidth);
    end
endfunction

% test handling of single sample per symbol phase jumps
function test_phase_est
  theta = 0:0.01:2*pi;
  phi_fine =angle((exp(j*theta)).^2)/2;
  phi_coarse = zeros(1,length(phi_fine));
  for n=2:length(phi_fine)
    phi_coarse(n) = phi_coarse(n-1);
    if phi_fine(n) - phi_fine(n-1) < -0.9*pi
      phi_coarse(n) += pi;
    end
    if phi_fine(n) - phi_fine(n-1) > 0.9*pi
      phi_coarse(n) -= pi;
    end
  end
  phi = phi_coarse + phi_fine;
  figure(1); clf; hold on;
  plot(theta, phi_fine, "b-;fine;")
  plot(theta, phi_coarse, "g-;coarse;")
  plot(theta, phi, "r-;phi;")
  hold off
endfunction

function compare_pitch_corr(wav_fn,feat1_fn,feat2_fn,png_feat_fn="")
  Fs=16000;
  s=load_raw(wav_fn);
  feat1=load_f32(feat1_fn,36);
  feat2=load_f32(feat2_fn,36);

  % plot a few features over first 5 seconds

  x_wav=(1:Fs*5);
  x=1:min(500,length(feat2));
  figure(1); clf;
  subplot(311);
  plot(s(x_wav),'g'); 
  axis([0 max(x_wav) -3E4 3E4]);
  subplot(312);
  plot(x,feat1(x,19),'g;fargan;');
  hold on;  plot(x,feat2(x,19),'r;radae;'); hold off;
  ylabel('Pitch'); axis([0 max(x) -1 1]);
  subplot(313);
  plot(x,feat1(x,20),'g;fargan;');
  hold on; plot(x,feat2(x,20),'r;radae;'); hold off;
  ylabel('Corr'); xlabel('10ms frames');
  axis([0 max(x) -0.6 0.6])
  if length(png_feat_fn)
    print("-dpng",png_feat_fn,"-S1200,800");
  end
end

function plot_sample_spec(wav_fn,png_spec_fn="")
  % plot spectrum of entire sample, to get a feel for input mic filtering

  Fs=16000;
  s=load_raw(wav_fn);
  figure(1); clf;
  S = 20*log10(abs(fft(s)(1:length(s)/2)));
  x_scale = (Fs/2)/length(S);
  semilogx((1:length(S))*x_scale,S)
  axis([1 8000 60 160]);
  xlabel('Frequency (Hz)'); ylabel('Amplitude (dB)');
  if length(png_spec_fn)
    print("-dpng",png_spec_fn,"-S800,600");
  end
end

function plot_ber_EbNodB(lin_fn,mse_fn="",phase_fn="",png="", epslatex="")
  if length(epslatex)
    [textfontsize linewidth] = set_fonts(20);
  end
  figure(1); clf;
  lin=load(lin_fn);
  semilogy(lin(:,1),lin(:,2),'bo-;lin;')
  hold on;
  if length(mse_fn)
    mse=load(mse_fn);
    semilogy(mse(:,1),mse(:,2),'g+-;mse;')
  end
  if length(phase_fn)
    phase=load(phase_fn);
    semilogy(phase(:,1),phase(:,2),'rx-;phase;')
  end
  EbNoLin = 10.^(lin(:,1)/10);
  theory = 0.5*erfc(sqrt(EbNoLin));
  semilogy(lin(:,1),theory,'bk+-;theory;')
  hold off;
  grid; xlabel('Eb/No (dB)'); ylabel('BER');
  axis([lin(1,1) lin(end,1) 1E-3 5E-1]);
  if length(png)
    print("-dpng",png,"-S800,600");
  end
  if length(epslatex)
    print_eps_restore(epslatex,"-S300,300",textfontsize,linewidth);
  end

end
