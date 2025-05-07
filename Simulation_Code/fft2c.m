function output = fft2c(input)

    L = size(input);
    
    output = input;
    n = L(1) * L(2);
    
    for ii=1:2
        output = ifftshift(fft(fftshift(output, ii),[], ii), ii);
    end
    
    output = (1/sqrt(n))*output; 