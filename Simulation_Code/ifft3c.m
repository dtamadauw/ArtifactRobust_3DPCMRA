function output = ifft3c(input, varargin)

    L = size(input);
    
    output = input;
    n = L(1) * L(2) * L(3);
    
    for ii=1:3
        output = ifftshift(ifft(fftshift(output, ii),[], ii), ii);
    end
    
    output = sqrt(n)*output;