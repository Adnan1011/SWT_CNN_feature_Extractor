% Author: Adnan Chaudhry
% Date created: April 22, 2017
%% Convolutional Autoencoder
% Code for training a convolutional autoencoder for determining initial
% weights for the shallow CNN operating on top of Hand crafted  features
% For STL 10 dataset
function script_CAE_HC_Feats_STL_10()
clc;
clearvars -except images num_images;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
active_caffe_mex(auto_select_gpu());

%% Script settings
dataset = fullfile(pwd, 'datasets', 'stl10_matlab', 'unlabeled');
solver_def_file = fullfile(pwd, 'models', 'CAE_STL_10_prototxts', 'solver.prototxt');
%weights_file = fullfile(pwd, 'output', 'CAE_cachedir', 'CAE_SWT_1_layer_80_filts_iter_4655.caffemodel');
weights_file = fullfile(pwd, 'output', 'CAE_STL_10_cachedir', 'CAE_final.caffemodel');
rng_seed = 7;
batch_size = 100;
snapshot_interval = 1000;
use_gpu = true;
% Spatial size of input image/feature map
input_size = [129 129];
copy_weights = true;

%% building dataset
% Export to / Import from base workspace to speed up loading when the 
% script is run multiple times
try
    images = evalin('base', 'images');
    num_images = evalin('base', 'num_images');
catch
    [images, num_images] = build_image_dataset(dataset);
end

%% init caffe solver       
cache_dir = fullfile(pwd, 'output', 'CAE_STL_10_cachedir');
mkdir_if_missing(cache_dir);
caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
caffe.init_log(caffe_log_file_base);
caffe_solver = caffe.Solver(solver_def_file);
% set random seed
prev_rng = seed_rand(rng_seed);
caffe.set_random_seed(rng_seed);
% set gpu/cpu
if use_gpu
    caffe.set_mode_gpu();
else
    caffe.set_mode_cpu();
end
shuffled_images = [];
if copy_weights   
    caffe_solver.net.copy_from(weights_file);
end
% initialize some params
epoch_size = ceil(num_images / batch_size);
max_iters = caffe_solver.max_iter();
iter = caffe_solver.iter();
% Current position in image files array. Used for generating mini batches
current_pos = 1;
% helpful for visualizing loss curves
training_results = [];
caffe_solver.net.set_phase('train');
% % intialize real time plotting of training loss
% figure_handle = figure('NumberTitle', 'off', 'Visible', 'off', ...
%         'Name', 'Training loss');
% axes_handle = axes('Parent',figure_handle, 'YGrid', 'on', 'XGrid', 'on');
% plot_handle=plot(axes_handle, 0, 0);

%% Training loop
while(iter < max_iters)
    if ~mod(iter, epoch_size)
        shuffled_images = images(randperm(num_images), :, :, :);
        current_pos = 1;
    end
    [mini_batch, current_pos] = get_next_mini_batch(shuffled_images, ...
        num_images, current_pos, batch_size);
    input_blob = get_input_blob(mini_batch, input_size, batch_size);
    caffe_solver.net.reshape_as_input(input_blob);    
    caffe_solver.net.set_input_data(input_blob);
    % Run one forward and backward pass
    caffe_solver.step(1);    
    rst = caffe_solver.net.get_output();    
    training_results = parse_rst(training_results, rst);
    display(['iter: ' num2str(iter) ' loss = ' num2str(training_results.cross_entropy_loss.data(iter + 1)) ...
        ', MSE = ' num2str(training_results.l2_error.data(iter + 1))]);
%     % plot training loss
%     set(plot_handle, 'YData', training_results.cross_entropy_loss.data', ...
%         'XData', 1 : (iter + 1));
%     set(figure_handle, 'Visible', 'on');
    % snapshot
    if (~mod(iter, snapshot_interval)) && (iter ~= 0)
        snapshot(caffe_solver, cache_dir, sprintf('CAE_iter_%d.caffemodel', iter));
    end
    iter = caffe_solver.iter();
end
%% Visualize training losses
figure;
plot(1:max_iters, training_results.cross_entropy_loss.data');
xlabel('iterations');
ylabel('cross entropy loss');
title('CAE cross entropy training loss');
figure;
plot(1:max_iters, training_results.l2_error.data');
xlabel('iterations');
ylabel('MSE');
title('CAE MSE vs iterations');
%% Finalize
% final snapshot
snapshot(caffe_solver, cache_dir, 'CAE_final.caffemodel');
caffe.reset_all();
% restore previous random number generator 
rng(prev_rng);

end

function [images, num_images] = build_image_dataset(dataset_mat)    
    ld = load(dataset_mat);
    images = ld.X;
    num_images = size(images, 1);
    images = reshape(images, num_images, 96, 96, 3);    
    clear 'ld'
    assignin('base', 'images', images);
    assignin('base', 'num_images', num_images);
end

function [mini_batch, current_pos] = get_next_mini_batch(images, num_images, ...
    current_pos, batch_size)    
    % subtract 1 to have zero indexed position
    adjusted_pos = current_pos - 1;
    % add back 1 to have 1 indexed positions
    mini_batch_inds = mod(adjusted_pos : 1 : adjusted_pos + batch_size - 1, ...
        num_images) + 1;
    mini_batch = images(mini_batch_inds, :, :, :);
    current_pos = current_pos + batch_size;
end

function snapshot(caffe_solver, cache_dir, file_name)    
    model_path = fullfile(cache_dir, file_name);
    caffe_solver.net.save(model_path);
    fprintf('Saved as %s\n', model_path);    
end

function input_blob = get_input_blob(mini_batch, input_size, batch_size)   
    features = cell(batch_size, 1);    
    for i = 1 : batch_size        
        feat_im = GenerateFeatures('', 'SWT', squeeze(mini_batch(i, :, :, :)));
        % resize
        feat_im = imresize(feat_im, input_size);
        features{i} = feat_im; 
    end    
    input_blob = im_list_to_blob(features);
    input_blob = single(permute(input_blob, [2, 1, 3, 4]));
    input_blob = {input_blob};
end
