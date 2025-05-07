%TITLE: MRI simulation code for pulsation artifacts
%Matlab script to generate Figures 4 in the paper
%Author: Daiki Tamada
%Affiliation: Department of Radiology, University of Wisconsin-Madison
%Date: 5/7/2025
%Email: dtamada@wisc.edu


%%
%Fixed flow mean
spect = ones(256,8,112);
ph_order = 2;

TR = 10.0e-3;
%Rosengarten, B., et al. "Comparison of visually evoked peak systolic and end diastolic blood flow velocity using a control system approach." Ultrasound in medicine & biology 27.11 (2001): 1499-1503.
flow_flac = 0:1:50;%cm/s
flow_mean = 0.0;

%Sapra et al., 2021 Sapra, A., Malik, A., Bhandari, P., 2021. Vital Sign Assessment, StatPearls, Treasure Island (FL).
N = 50;

pslr_venc10 = zeros(length(flow_flac),N);
pslr_venc30 = zeros(length(flow_flac),N);

for n = 1:N
    freq = unifrnd(1.0,1.667);%Hz
    
    venc = 10;
    for ii=1:length(flow_flac)
            [spect_error, phaseError] = addErrors(spect, flow_flac(ii), freq, flow_mean, TR, venc, ph_order);
            pslr_venc10(ii,n) = calculate_pslr(ifft3c(spect_error));
    end
    venc = 30;
    for ii=1:length(flow_flac)
            [spect_error, phaseError] = addErrors(spect, flow_flac(ii), freq, flow_mean, TR, venc, ph_order);
            pslr_venc30(ii,n) = calculate_pslr(ifft3c(spect_error));
    end
end

%%

pslr_venc10_mean = mean(pslr_venc10, 2);
pslr_venc30_mean = mean(pslr_venc30, 2);

figure;
plot(flow_flac, pslr_venc10_mean, 'LineWidth', 2); 
xlabel('mean-to-peak velocity (cm/s)'); ylabel('PSLR (dB)');
hold on;
plot(flow_flac, pslr_venc30_mean, 'LineWidth', 2);
xlabel('Mean-to-Peak Velocity (cm/s)'); ylabel('PSLR (dB)');
set(gca, 'fontname', 'Arial', 'FontSize',14,'FontWeight','normal','LineWidth',2);
xlim([1 50]); ylim([0 80]);
legend({'VENC=10cm/s', 'VENC=30cm/s'});legend('boxoff');
set(gcf,'units','inches','position',[0,0,5,5]);

%%
%SL->PE ordering

ph_order = 0;
venc = 30;
error_log_10 = zeros(256, 112, 50);
error_log_20 = zeros(256, 112, 50);
error_log_30 = zeros(256, 112, 50);

for ii=1:100
    freq = 1.0;%unifrnd(1.0,1.667);%Hz
    [spect_error, phaseError] = addErrors(spect, 10, freq, flow_mean, TR, venc, ph_order);
    error_img = ifft3c(spect_error);
    error_img_10(:,:,ii) = squeeze(error_img(:,5,:));
    [spect_error, phaseError] = addErrors(spect, 20, freq, flow_mean, TR, venc, ph_order);
    error_img = ifft3c(spect_error);
    error_img_20(:,:,ii) = squeeze(error_img(:,5,:));
    [spect_error, phaseError] = addErrors(spect, 30, freq, flow_mean, TR, venc, ph_order);
    error_img = ifft3c(spect_error);
    error_img_30(:,:,ii) = squeeze(error_img(:,5,:));
end

error_log = abs(mean(error_img_10,3)); 
error_log_filt = imgaussfilt(error_log, 0.5); max_error = max(error_log_filt(:));
error_log_filt = (error_log_filt/max_error);
figure; semilogy(error_log_filt(:,57), 'LineWidth', 2);



error_log = abs(sum(error_img_20,3)); 
error_log_filt = imgaussfilt(error_log, 0.5); max_error = max(error_log_filt(:));
error_log_filt = (error_log_filt/max_error);
hold on; semilogy(error_log_filt(:,57), 'LineWidth', 2);

error_log = abs(sum(error_img_30,3)); 
error_log_filt = imgaussfilt(error_log, 0.5); max_error = max(error_log_filt(:));
error_log_filt = (error_log_filt/max_error);
hold on; semilogy(error_log_filt(:,57), 'LineWidth', 2);
legend({'10cm/s', '20cm/s', '30cm/s'});legend('boxoff');
xlabel('PE');ylabel('Normalized Intensity');
set(gca, 'fontname', 'Arial', 'FontSize',14,'FontWeight','normal','LineWidth',2);
set(gcf,'units','inches','position',[0,0,5,5]);xlim([1 256])
grid on;



%%
%PE->SL ordering

ph_order = 1;
venc = 30;
error_log_10 = zeros(256, 112, 50);
error_log_20 = zeros(256, 112, 50);
error_log_30 = zeros(256, 112, 50);

for ii=1:100
    freq = 1.0;%unifrnd(1.0,1.667);%Hz
    [spect_error, phaseError] = addErrors(spect, 10, freq, flow_mean, TR, venc, ph_order);
    error_img = ifft3c(spect_error);
    error_img_10(:,:,ii) = squeeze(error_img(:,5,:));
    [spect_error, phaseError] = addErrors(spect, 20, freq, flow_mean, TR, venc, ph_order);
    error_img = ifft3c(spect_error);
    error_img_20(:,:,ii) = squeeze(error_img(:,5,:));
    [spect_error, phaseError] = addErrors(spect, 30, freq, flow_mean, TR, venc, ph_order);
    error_img = ifft3c(spect_error);
    error_img_30(:,:,ii) = squeeze(error_img(:,5,:));
end

error_log = abs(mean(error_img_10,3)); 
error_log_filt = imgaussfilt(error_log, 0.5); max_error = max(error_log_filt(:));
error_log_filt = (error_log_filt/max_error);
figure; semilogy(error_log_filt(:,57), 'LineWidth', 2);



error_log = abs(sum(error_img_20,3)); 
error_log_filt = imgaussfilt(error_log, 0.5); max_error = max(error_log_filt(:));
error_log_filt = (error_log_filt/max_error);
hold on; semilogy(error_log_filt(:,57), 'LineWidth', 2);

error_log = abs(sum(error_img_30,3)); 
error_log_filt = imgaussfilt(error_log, 0.5); max_error = max(error_log_filt(:));
error_log_filt = (error_log_filt/max_error);
hold on; semilogy(error_log_filt(:,57), 'LineWidth', 2);
legend({'10cm/s', '20cm/s', '30cm/s'});legend('boxoff');
xlabel('PE');ylabel('Normalized Intensity');
set(gca, 'fontname', 'Arial', 'FontSize',14,'FontWeight','normal','LineWidth',2);
set(gcf,'units','inches','position',[0,0,5,5]);xlim([1 256])
grid on;



%%
%RL flipping

ph_order = 2;
venc = 30;
error_log_10 = zeros(256, 112, 50);
error_log_20 = zeros(256, 112, 50);
error_log_30 = zeros(256, 112, 50);

for ii=1:100
    freq = 1.0;%unifrnd(1.0,1.667);%Hz
    [spect_error, phaseError] = addErrors(spect, 10, freq, flow_mean, TR, venc, ph_order);
    error_img = ifft3c(spect_error);
    error_img_10(:,:,ii) = squeeze(error_img(:,5,:));
    [spect_error, phaseError] = addErrors(spect, 20, freq, flow_mean, TR, venc, ph_order);
    error_img = ifft3c(spect_error);
    error_img_20(:,:,ii) = squeeze(error_img(:,5,:));
    [spect_error, phaseError] = addErrors(spect, 30, freq, flow_mean, TR, venc, ph_order);
    error_img = ifft3c(spect_error);
    error_img_30(:,:,ii) = squeeze(error_img(:,5,:));
end

error_log = abs(mean(error_img_10,3)); 
error_log_filt = imgaussfilt(error_log, 0.5); max_error = max(error_log_filt(:));
error_log_filt = (error_log_filt/max_error);
figure; semilogy(error_log_filt(:,57), 'LineWidth', 2);



error_log = abs(sum(error_img_20,3)); 
error_log_filt = imgaussfilt(error_log, 0.5); max_error = max(error_log_filt(:));
error_log_filt = (error_log_filt/max_error);
hold on; semilogy(error_log_filt(:,57), 'LineWidth', 2);

error_log = abs(sum(error_img_30,3)); 
error_log_filt = imgaussfilt(error_log, 0.5); max_error = max(error_log_filt(:));
error_log_filt = (error_log_filt/max_error);
hold on; semilogy(error_log_filt(:,57), 'LineWidth', 2);
legend({'10cm/s', '20cm/s', '30cm/s'});legend('boxoff');
xlabel('PE');ylabel('Normalized Intensity');
set(gca, 'fontname', 'Arial', 'FontSize',14,'FontWeight','normal','LineWidth',2);
set(gcf,'units','inches','position',[0,0,5,5]);xlim([1 256])
grid on;


