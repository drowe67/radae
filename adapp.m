% adapp.m
% Post processing of adasmooth delays to avoid MPP wandering that upsets network

function dpp = adapp(d)
    dpp = zeros(length(d),1);
    count = 0;
    beta = 0.99;
    for n=2:length(d)
      if abs(d(n)-dpp(n-1)) > 16
        count++;
      end
      if count > 5
        dpp(n) = d(n);
        count = 0;
      else
        dpp(n) = dpp(n-1)*beta + d(n)*(1-beta);
      end
	end
end
