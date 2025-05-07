function output = ifft2c(input, varargin)
    
    L = size(input);
    
    output = input;
    n = L(1) * L(2);
    
    for ii=1:2
        output = ifftshift(ifft(fftshift(output, ii),[], ii), ii);
    end
    
    output = sqrt(n)*output; 