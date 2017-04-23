% Author: Adnan Chaudhry
% Date created: April 22, 2017
%% Convolutional Autoencoder
% Code for training a convolutional autoencoder for determining initial
% weights for the shallow CNN operating on top of Hand crafted  features
% For STL 10 dataset
function script_CLS_HC_Feats_STL_10()
clc;
clearvars -except images num_images;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
active_caffe_mex(auto_select_gpu());

%% Script settings
dataset = fullfile(pwd, 'datasets', 'stl10_matlab', 'train');
solver_def_file = fullfile(pwd, 'models', 'CLS_STL_10_prototxts', 'solver.prototxt');
weights_file = fullfile(pwd, 'output', 'CAE_STL_10_cachedir', 'CAE_final.caffemodel');
%weights_file = fullfile(pwd, 'output', 'CLS_STL_10_cachedir', 'CLS_STL_10_48.61%.caffemodel');
rng_seed = 7;
batch_size = 50;
snapshot_interval = 1000;
use_gpu = true;
copy_weights = true;

%% building dataset
[images, num_images, labels] = build_image_dataset(dataset);

%% init caffe solver       
cache_dir = fullfile(pwd, 'output', 'CLS_STL_10_cachedir');
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
shuffled_labels = [];
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

%% Training loop
while(iter < max_iters)
    if ~mod(iter, epoch_size)
        shuffled_inds = randperm(num_images);
        shuffled_images = images(shuffled_inds, :, :, :);
        shuffled_labels = labels(shuffled_inds);
        current_pos = 1;
    end
    [mini_batch_inds, current_pos] = get_next_mini_batch_inds(shuffled_inds, ...
        num_images, current_pos, batch_size);
    input_blob = get_input_blob(mini_batch_inds, shuffled_images, ...
        shuffled_labels, batch_size);
    caffe_solver.net.reshape_as_input(input_blob);    
    caffe_solver.net.set_input_data(input_blob);
    % Run one forward and backward pass
    caffe_solver.step(1);    
    rst = caffe_solver.net.get_output();    
    training_results = parse_rst(training_results, rst);
    display(['iter: ' num2str(iter) ' loss = ' num2str(training_results.softmax_loss.data(iter + 1)) ...
        ', accuracy = ' num2str(training_results.accuracy.data(iter + 1))]);
    % snapshot
    if (~mod(iter, snapshot_interval)) && (iter ~= 0)
        snapshot(caffe_solver, cache_dir, sprintf('CLS_STL_10_iter_%d.caffemodel', iter));
    end
    if (~mod(iter, 7999)) && (iter ~= 0)
        display(['Test accuracy: ' num2str(mean(training_results.accuracy.data))]);
    end
    iter = caffe_solver.iter();
end
%% Visualize training losses
figure;
plot(1:max_iters, training_results.softmax_loss.data');
xlabel('iterations');
ylabel('Softmax loss');
title('CLS softmax training loss');
figure;
plot(1:max_iters, training_results.accuracy.data');
xlabel('iterations');
ylabel('Accuracy');
title('Classification accuracy vs iterations');
%% Finalize
% final snapshot
snapshot(caffe_solver, cache_dir, 'CLS_STL_10_final.caffemodel');
caffe.reset_all();
% restore previous random number generator 
rng(prev_rng);

end

function [images, num_images, labels] = build_image_dataset(dataset_mat)    
    ld = load(dataset_mat);
    images = ld.X;
    num_images = size(images, 1);
    images = reshape(images, num_images, 96, 96, 3);
    labels = ld.y';    
    clear 'ld'    
end

function [mini_batch_inds, current_pos] = get_next_mini_batch_inds(inds, ...
    num_images, current_pos, batch_size)    
    % subtract 1 to have zero indexed position
    adjusted_pos = current_pos - 1;
    % add back 1 to have 1 indexed positions
    mini_batch_inds = mod(adjusted_pos : 1 : adjusted_pos + batch_size - 1, ...
        num_images) + 1;
    mini_batch_inds = inds(mini_batch_inds);
    current_pos = current_pos + batch_size;
end

function snapshot(caffe_solver, cache_dir, file_name)    
    model_path = fullfile(cache_dir, file_name);
    caffe_solver.net.save(model_path);
    fprintf('Saved as %s\n', model_path);    
end

function input_blob = get_input_blob(mini_batch_inds, images, labels, ...
    batch_size)   
    features = cell(batch_size, 1);    
    for i = 1 : batch_size        
        feat_im = GenerateFeatures('', 'SWT', ...
            squeeze(images(mini_batch_inds(i), :, :, :)));        
        features{i} = feat_im; 
    end    
    feat_blob = im_list_to_blob(features);
    feat_blob = single(permute(feat_blob, [2, 1, 3, 4]));
    label_blob = labels(mini_batch_inds);
    input_blob = {feat_blob, label_blob};
end
