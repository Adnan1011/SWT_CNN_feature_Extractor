% Author: Adnan Chaudhry
% Date created: April 8, 2017
%% Convolutional Autoencoder
% Code for training a convolutional autoencoder for determining initial
% weights for the shallow CNN operating on top of Hand crafted  features
function script_CAE_HC_Feats()
clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
active_caffe_mex(auto_select_gpu());

%% Script settings
image_set_path = fullfile(pwd, 'datasets', 'VOCdevkit2007', 'VOC2007', ...
    'ImageSets', 'Main');
image_files_path = fullfile(pwd, 'datasets', 'VOCdevkit2007', 'VOC2007', ...
    'JPEGImages');
image_set_name = 'trainval.txt';
image_ext = '.jpg';
solver_def_file = fullfile(pwd, 'models', 'CAE_prototxts', 'solver.prototxt');
rng_seed = 7;
batch_size = 4;
snapshot_interval = 10000;
% Spatial size of input image/feature map
input_size = [127 127];
use_gpu = true;

%% image paths

[image_files, num_images] = build_image_dataset(image_set_path, image_set_name, ...
    image_files_path, image_ext);

%% init caffe solver       
cache_dir = fullfile(pwd, 'output', 'CAE_cachedir');
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
shuffled_img_files = [];
% initialize some params
epoch_size = ceil(num_images / batch_size);
max_iters = caffe_solver.max_iter();
iter = caffe_solver.iter();
% Current position in image files array. Used for generating mini batches
current_pos = 1;
% helpful for visualizing loss curves
training_results = [];

caffe_solver.net.set_phase('train');

%% Training loop
while(iter < max_iters)
    if ~mod(iter, epoch_size)
        shuffled_img_files = image_files(randperm(num_images));
        current_pos = 1;
    end
    [mini_batch, current_pos] = get_next_mini_batch(shuffled_img_files, ...
        current_pos, batch_size);
    input_blob = get_input_blob(mini_batch, input_size);
    caffe_solver.net.reshape_as_input(input_blob);    
    caffe_solver.net.set_input_data(input_blob);
    % Run one forward and backward pass
    caffe_solver.step(1);    
    rst = caffe_solver.net.get_output();    
    training_results = parse_rst(training_results, rst);
    % snapshot
    if ~mod(iter, snapshot_interval)
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
ylabel('L2 loss');
title('CAE L2 training loss');
%% Finalize
% final snapshot
snapshot(caffe_solver, cache_dir, 'CAE_final.caffemodel');
caffe.reset_all();
% restore previous random number generator 
rng(prev_rng);

end

function [image_files, num_images] = build_image_dataset(image_set_path, ...
    image_set_name, image_files_path, image_ext)
    image_set_file = fopen(fullfile(image_set_path, image_set_name), 'r');
    image_ids = textscan(image_set_file, '%s');
    fclose(image_set_file);
    image_ids = image_ids{1,1};
    num_images = length(image_ids);
    image_files = strcat(image_files_path, filesep, image_ids, image_ext);
end

function [mini_batch, current_pos] = get_next_mini_batch(image_files, ...
    current_pos, batch_size)
    num_images = length(image_files);
    % subtract 1 to have zero indexed position
    adjusted_pos = current_pos - 1;
    % add back 1 to have 1 indexed positions
    mini_batch_inds = mod(adjusted_pos : 1 : adjusted_pos + batch_size - 1, ...
        num_images) + 1;
    mini_batch = image_files(mini_batch_inds);
    current_pos = current_pos + batch_size;
end

function snapshot(caffe_solver, cache_dir, file_name)    
    model_path = fullfile(cache_dir, file_name);
    caffe_solver.net.save(model_path);
    fprintf('Saved as %s\n', model_path);    
end

function input_blob = get_input_blob(mini_batch, input_size)
    batch_size = length(mini_batch);
    features = cell(batch_size, 1);    
    for i = 1 : batch_size        
        feat_im = GenerateFeatures(mini_batch{i}, 'SWT');
        min_feature_val = min(min(feat_im));
        max_feature_val = max(max(feat_im));
        feat_im = (feat_im - min_feature_val) / (max_feature_val - min_feature_val);
        % resize
        feat_im = imresize(feat_im, input_size);
        features{i} = feat_im; 
    end    
    input_blob = im_list_to_blob(features);
    input_blob = single(permute(input_blob, [2, 1, 3, 4]));
    input_blob = {input_blob};
end
