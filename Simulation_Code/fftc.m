function output = fftc(input,varargin)


    L = size(input);
    
    output = input;
    n = L(1);
    
    if length(varargin) > 0
        ii = varargin{1};
    else
        ii=1;
     end
     output = ifftshift(fft(fftshift(output, ii),[], ii), ii);
  
    
    output = (1/sqrt(n))*output; 
