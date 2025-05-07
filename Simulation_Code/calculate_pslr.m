function [pslr, details] = calculate_pslr(psf3D, options)
    % calculatePSLR - Calculate Peak-to-Side-Lobe Ratio from 3D PSF
    %
    % Syntax:
    %   pslr = calculatePSLR(psf3D)
    %   [pslr, details] = calculatePSLR(psf3D)
    %   [pslr, details] = calculatePSLR(psf3D, options)
    %
    % Inputs:
    %   psf3D - 3D array containing the Point Spread Function
    %   options - (optional) Structure with the following fields:
    %       .mainLobeRadius - Radius for main lobe as fraction of smallest dimension (default: 0.1)
    %       .useAbsoluteValues - Whether to use absolute values of the PSF (default: false)
    %       .mainLobeCenterMethod - Method to find the main lobe center:
    %           'geometricCenter' (default) - Use the geometric center of the array
    %           'maxValue' - Use the position of the maximum value
    
    % Default options
    defaultOptions.mainLobeRadius = 0.1;
    defaultOptions.useAbsoluteValues = false;
    defaultOptions.mainLobeCenterMethod = 'maxValue';
    
    % Use default options if not provided
    if nargin < 2
        options = defaultOptions;
    else
        % Fill in missing options with defaults
        fieldNames = fieldnames(defaultOptions);
        for i = 1:length(fieldNames)
            if ~isfield(options, fieldNames{i})
                options.(fieldNames{i}) = defaultOptions.(fieldNames{i});
            end
        end
    end
    
    % Get the dimensions of the PSF
    [sizeX, sizeY, sizeZ] = size(psf3D);
    
    % Use absolute values if requested
    %if options.useAbsoluteValues
        psf3D = abs(psf3D);
    %end
    
    % Find the center of the PSF based on the specified method
    switch options.mainLobeCenterMethod
        case 'geometricCenter'
            centerX = ceil(sizeX/2);
            centerY = ceil(sizeY/2);
            centerZ = ceil(sizeZ/2);
            mainLobePeak = psf3D(centerX, centerY, centerZ);
        case 'maxValue'
            [mainLobePeak, maxIdx] = max(psf3D(:));
            [centerX, centerY, centerZ] = ind2sub(size(psf3D), maxIdx);
        otherwise
            error('Unknown main lobe center method: %s', options.mainLobeCenterMethod);
    end
    
    % Create coordinate grids for the entire volume
    [Y, X, Z] = meshgrid(1:sizeY, 1:sizeX, 1:sizeZ);
    
    % Calculate distance from center for all points
    distances = sqrt((X-centerX).^2 + (Y-centerY).^2 + (Z-centerZ).^2);
    
    % Create a mask for the main lobe region
    radius = floor(min([sizeX, sizeY, sizeZ]) * options.mainLobeRadius);
    mainLobeMask = distances <= radius;
    
    % Create a copy of the PSF with the main lobe set to zero
    psf3D_noMainLobe = psf3D;
    psf3D_noMainLobe(mainLobeMask) = 0;
    
    % Find the maximum value in the side lobes
    [maxSideLobe, maxSideLobeIdx] = max(psf3D_noMainLobe(:));
    [maxSideLobeX, maxSideLobeY, maxSideLobeZ] = ind2sub(size(psf3D), maxSideLobeIdx);
    
    % Calculate the PSLR in linear scale
    pslrLinear = mainLobePeak / maxSideLobe;
    
    % Convert to dB
    pslr = 20 * log10(pslrLinear);
    
    % Prepare detailed output if requested
    if nargout > 1
        details.mainLobePeak = mainLobePeak;
        details.maxSideLobe = maxSideLobe;
        details.mainLobeCenter = [centerX, centerY, centerZ];
        details.maxSideLobePos = [maxSideLobeX, maxSideLobeY, maxSideLobeZ];
        details.pslrLinear = pslrLinear;
        details.mainLobeRadius = radius;
    end
end


%{
function pslr = calculate_pslr(psf)
    % CALCULATE_PSLR Calculates Peak-to-Side-Lobe Ratio from 3D PSF
    %   psf: Input 3D point spread function array
    %   pslr: Peak-to-Side-Lobe Ratio in dB
    
    % Ensure PSF is in magnitude form
    psf_mag = abs(psf);
    
    % Normalize PSF
    psf_norm = psf_mag / max(psf_mag(:));
    
    % Find the main peak location
    [~, peak_idx] = max(psf_norm(:));
    [peak_i, peak_j, peak_k] = ind2sub(size(psf_norm), peak_idx);
    
    % Create 3D mask to exclude main lobe
    % Assuming main lobe width is approximately 3 voxels
    main_lobe_width = 5;
    [X, Y, Z] = meshgrid(1:size(psf_norm,2), 1:size(psf_norm,1), 1:size(psf_norm,3));
    main_lobe_mask = (X - peak_j).^2 + (Y - peak_i).^2 + (Z - peak_k).^2 <= (main_lobe_width/2)^2;
    
    % Mask out the main lobe
    side_lobes = psf_norm;
    side_lobes(main_lobe_mask) = 0;
    
    % Find maximum side lobe value
    max_side_lobe = max(side_lobes(:));
    
    % Calculate PSLR in dB
    pslr = 20 * log10(1/max_side_lobe); % Main peak is 1 due to normalization
    
end

%}