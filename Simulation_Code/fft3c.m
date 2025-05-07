function output = fft3c(input, varargin)
    
    
    L = size(input);
    
    output = input;
    n = L(1) * L(2) * L(3);
    
    for ii=1:3
        output = ifftshift(fft(fftshift(output, ii),[], ii), ii);
    end
    
    output = (1/sqrt(n))*output;
    