function [spect, phaseError] = addErrors(spect, A, f, B, TR, venc,ph_order)

    arguments
        spect (:,:,:) double
        A (:,1) double
        f (1,1) double
        B (1,1) double
        TR (1,1) double
        venc (1,1) double
        ph_order (1,1) int16 = 1

    end


    % Get the size of the input spect array.
    shape = size(spect); % shape(1)=dim1, shape(2)=dim2, shape(3)=dim3
    %disp(shape);
    
    % Calculate duration and systolic duration (with random error)
    hrv_sd = 42e-3;
    duration = 1 / f;
    durations = duration + (hrv_sd *randn(1,shape(1)*shape(3)));
    duration_systolic = 0.38*duration;%(360.0 + 0*(rand * 58 - 29)) * 1e-3;
    
    % Create grid for phase indexing.
    % Here, x corresponds to columns (second dimension) and
    % y corresponds to pages (third dimension).
    x = linspace(0, shape(1)-1, shape(1));
    y = linspace(0, shape(3)-1, shape(3)); 
    [xv, yv] = meshgrid(x, y);  % xv and yv are matrices of size [shape(3) x shape(2)]
    xv = xv-shape(1)/2; yv = yv - shape(3)/2;
    a = shape(1)/2; b = shape(3)/2;
    rv = sqrt((xv/a).^2+(yv/b).^2);
    ellip_mask = abs(rv)<0.8;
    phaseIndex = abs(rv); pind = 1;
    

    % Compute phase index based on the number of TRs (total grid of indices).
    if ph_order == 1
        for ii=1:shape(3)
            for jj=1:shape(1)
                if ellip_mask(ii,jj)
                    phaseIndex(ii,jj) = pind;
                    pind = pind+1;
                else 
                    phaseIndex(ii,jj) = 0;
                end
            end
        end
        %phaseIndex = yv * shape(3) + xv;%PE->SL
    elseif ph_order == 0
        for jj=1:shape(1)
            for ii=1:shape(3)
                if ellip_mask(ii,jj)
                    phaseIndex(ii,jj) = pind;
                    pind = pind+1;
                else 
                    phaseIndex(ii,jj) = 0;
                end
            end
        end
        %phaseIndex = xv * shape(1) + yv;%SL->PE
    elseif ph_order == 2
        for jj=shape(1):-1:1
            for ii=shape(3):-1:1
                if ellip_mask(ii,jj)
                    phaseIndex(ii,jj) = pind;
                    pind = pind+1;
                else 
                    phaseIndex(ii,jj) = 0;
                end
            end
        end
        %[xv, yv] = meshgrid(x(end:-1:1), y); 
        %phaseIndex = xv * shape(1) + yv;%SL->reversed PE
    end
    
    % Compute time for each index, and wrap by duration.
    t_mat = phaseIndex * TR;
    t_ind = 1;
    for ii=1:length(durations)
        max_t = floor(durations(ii)/TR);
        if (t_ind+max_t-1) > numel(t_mat)
            t_mat(t_ind:end)=0;
        else
            t_mat(t_ind:(t_ind+max_t-1)) = TR*(1:max_t);
            t_ind = t_ind+max_t;
        end
    end
    t_mat(t_mat > duration_systolic) = 0.0;
    
    % Evaluate the sinusoidal wave at each time point.
    % Note: Using 1/duration_systolic as the frequency.
    velocity_at_t = A .* sin(pi * (1/duration_systolic) .* t_mat') + B;  
    % ----- Mask out center of k-space -----
    % We build a 3D grid based on the dimensions of spect.
    s0 = shape(1);
    s1 = shape(2);
    s2 = shape(3);
    x_vec = linspace(-s0/2, s0/2 - 1, s0);
    y_vec = linspace(-s1/2, s1/2 - 1, s1);
    z_vec = linspace(-s2/2, s2/2 - 1, s2);
    
    % Use meshgrid to form 3D arrays. Note that MATLAB's meshgrid
    % returns matrices with first dimension corresponding to y and second to x.
    % This ordering is acceptable here because we use only y and z for the mask.
    [xv_k, yv_k, zv_k] = meshgrid(y_vec, x_vec, z_vec);
    
    % Create the mask based on the condition sqrt(y^2 + 4*z^2) > 10.
    vsz = sqrt(yv_k.^2 + 4 * zv_k.^2);
    vsz = double(vsz > 3);

    
    % Compute phase error.
    % It is assumed that the function phase_error_model is defined separately.
    phaseError = pi*velocity_at_t./venc + pi/2;    

    % Reshape phaseError from a vector (or matrix) into a 3D array of size [s0,1,s2].
    phaseError = reshape(phaseError, [s0, 1, s2]);
    % Replicate it along the second dimension.
    phaseError = repmat(phaseError, [1, s1, 1]);
    
    % Combine the phase errors with the mask.
    phaseError = -1i * sinh( phaseError * 1i ) .* vsz + (1 - vsz);
    
    % Apply the phase error to the spect data.
    spect = spect .* phaseError;
end
