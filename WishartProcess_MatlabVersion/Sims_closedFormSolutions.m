clear all; close all; clc; rng(1);
x_ref = [0,0];
x_comp1 = x_ref;
x_comp2 = [-4,0];

sig_ref_temp = randn(2);
sig_ref = sig_ref_temp*sig_ref_temp';
sig_comp1_temp = rand(2);
sig_comp1 = sig_ref;
sig_comp2_temp = rand(2);
sig_comp2 = sig_ref;

numSims = 1e3;
z_ref = mvnrnd(x_ref, sig_ref, numSims);
z_comp1 = mvnrnd(x_comp1, sig_comp1, numSims);
z_comp2 = mvnrnd(x_comp2, sig_comp2, numSims);

%visualize it
figure
scatter(z_ref(:,1), z_ref(:,2),'ro'); hold on
scatter(z_comp1(:,1), z_comp1(:,2),'bo');
scatter(z_comp2(:,1), z_comp2(:,2),'go');

%% compute the probability of correct responses
nGrid = 28;
z_grid_dim1 = linspace(-2,2,nGrid); z_grid_dim2 = linspace(-15,15,nGrid);
[Z1, Z2] = meshgrid(z_grid_dim1, z_grid_dim2);
[p_ref_comp1, p_ref_comp2] = deal(NaN(numSims, nGrid, nGrid));
for n = 1:numSims
    for g1 = 1:nGrid
        for g2 = 1:nGrid
            p_ref_comp1(n,g1,g2) = mvnpdf(z_ref(n,:), [Z1(g1,g2), Z2(g1,g2)],...
                sig_ref).*mvnpdf(z_comp1(n,:), [Z1(g1,g2), Z2(g1,g2)],...
                sig_comp1);
            p_ref_comp2(n,g1,g2) = mvnpdf(z_ref(n,:), [Z1(g1,g2), Z2(g1,g2)],...
                sig_ref).*mvnpdf(z_comp2(n,:), [Z1(g1,g2), Z2(g1,g2)],...
                sig_comp2);       
        end
    end
end
p_reporting_ref_comp1 = sum(sum(sum(p_ref_comp1,2),3) > sum(sum(p_ref_comp2,2),3))/numSims

%% moment generating function and inverse Fourier transformation
MG_func_chi = @(t, d, l) (1-2.*t).^(-d/2) .* exp(l.*t./(1-2.*t));
char_func_chi = @(t, d, l) exp((-i*l.*t)./(1-2*i.*t)) .* (1-2*i.*t).^(-d/2);

%noncentricity parameter
lambda_b = 0;
lambda_c = 4;
x_max    = 40;
N        = 1024;
xx       = linspace(-x_max,x_max,N); dx = abs(diff(xx(1:2)));
t        = 1:100;
%degree of freedom
d        = 4;
%noncentral chi square distribution
chi_b = ncx2pdf(xx, d, lambda_b);
chi_c = ncx2pdf(xx, d, lambda_c);

%sample from the two noncentral chi square distribution and visualize the
%difference between the two random variables
samples_chi_b = ncx2rnd(d, lambda_b,[1,1e5]);
samples_chi_c = ncx2rnd(d, lambda_c,[1,1e5]);
samples_bminusc = samples_chi_b - samples_chi_c;
figure; histogram(samples_bminusc,xx,'Normalization','probability','FaceAlpha',0.5);

%compute moment generating functions
MG_b        = arrayfun(@(idx) MG_func_chi(t(idx), d, lambda_b), t);
MG_c_minust = arrayfun(@(idx) MG_func_chi(-t(idx), d, lambda_c), t);
MG_bminusc = MG_b.*MG_c_minust;
figure; plot(MG_b,'r-'); hold on; plot(MG_bminusc, 'k-')

%compute characteristic functions
char_b        = arrayfun(@(idx) char_func_chi(t(idx), d, lambda_b), t);
char_c_minust = arrayfun(@(idx) char_func_chi(-t(idx), d, lambda_c), t);
char_bminusc = char_b.*char_c_minust;
figure; plot(t, abs(char_b),'r-'); hold on; plot(t, abs(char_bminusc),'k-')

%try to recover the probability density function
MG_bminusc_recovered = ifft(ifftshift(char_bminusc)); % Use ifftshift to align
MG_bminusc_recovered = real(MG_bminusc_recovered); % Take the real part
MG_bminusc_recovered = MG_bminusc_recovered / dt / (2 * pi); % Scale (dt and 2pi for Fourier Transform)
MG_bminusc_recovered = fftshift(MG_bminusc_recovered); % Shift zero frequency to the center
t_values = (-N/2:N/2-1) * (2*t_max/N); % Corresponding t values

figure; plot(t_values, abs(MG_bminusc_recovered));






