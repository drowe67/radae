function [y,d] = adasmooth(x)
	alpha = [0.2, 0.5, 0.75, 0.85, 0.92, 0.95, 0.98];
	h = zeros(size(x, 1), 1);
	y = x;
	for k=1:length(alpha)
	  y1 = filter(1-alpha(k), [1, -alpha(k)], x);
	  h1=(max(abs(y1'))./median(abs(y1')))';
	  mask = h1>h;
	  y = mask.*y1 + (1-mask).*y;
	  h = mask.*h1 + (1-mask).*h;
	end
	[a, d] = max(abs(y'));
end
