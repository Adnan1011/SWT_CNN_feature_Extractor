function script_faster_rcnn_HC_Feats_demo()
close all;
clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

opts.per_nms_topN           = 6000;
opts.nms_overlap_thres      = 0.7;
opts.after_nms_topN         = 300;
opts.use_gpu                = true;

opts.test_scales            = 600;
feature_depth               = 24;

%% -------------------- INIT_MODEL --------------------
model_dir                   = fullfile(pwd, 'models', 'trained_models'); %% HC Feats
proposal_detection_model    = load_proposal_detection_model(model_dir);
if opts.use_gpu
    proposal_detection_model.conf_proposal.image_means = gpuArray(proposal_detection_model.conf_proposal.image_means);
    proposal_detection_model.conf_detection.image_means = gpuArray(proposal_detection_model.conf_detection.image_means);
end
proposal_detection_model.conf_proposal.test_scales = opts.test_scales;
proposal_detection_model.conf_detection.test_scales = opts.test_scales;

% caffe.init_log(fullfile(pwd, 'caffe_log'));
% proposal net
rpn_net = caffe.Net(proposal_detection_model.proposal_net_def, 'test');
rpn_net.copy_from(proposal_detection_model.proposal_net);
% fast rcnn net
fast_rcnn_net = caffe.Net(proposal_detection_model.detection_net_def, 'test');
fast_rcnn_net.copy_from(proposal_detection_model.detection_net);

%conv_weights = rpn_net.params('conv1', 1).get_data();
% for i=1:size(conv_weights,3)
%         imshow(conv_weights(:,:,i),[])
%         title(num2str(i));
%         pause
% end
% set gpu/cpu
 if opts.use_gpu
    caffe.set_mode_gpu();
else
    caffe.set_mode_cpu();
end       

%% -------------------- WARM UP --------------------
% the first run will be slower; use an empty image to warm up

for j = 1:2 % we warm up 2 times
    im = uint8(ones(375, 500, feature_depth)*128);
%     if opts.use_gpu
%         im = gpuArray(im);
%     end
    [boxes, scores]             = proposal_im_detect(proposal_detection_model.conf_proposal, rpn_net, im, true);
    aboxes                      = boxes_filter([boxes, scores], opts.per_nms_topN, opts.nms_overlap_thres, opts.after_nms_topN, opts.use_gpu);
    
    [boxes, scores]             = fast_rcnn_im_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
                                  aboxes(:, 1:4), opts.after_nms_topN, true);
    
end

%% -------------------- TESTING --------------------
%im_names = {'001763.jpg', '004545.jpg', '000542.jpg', '000456.jpg', '001150.jpg'};
im_dir = strcat(pwd, '/datasets/VOCdevkit2007/train_val_images');
im_names = dir(fullfile(im_dir, '*.jpg'));
% these images can be downloaded with fetch_faster_rcnn_final_model.m

running_time = [];
for j = 1:length(im_names)    
    
    %im = GenerateFeatures(fullfile(pwd, im_names{j}), 'SWT');
    im = GenerateFeatures(fullfile(im_dir, im_names(j).name), 'SWT');
%     for i=1:size(im,3)
%         imshow(im(:,:,i),[])
%         title(num2str(i));
%         pause
%     end
%     if opts.use_gpu
%         im = gpuArray(im);
%     end
    
    % test proposal
    th = tic();
    [boxes, scores]             = proposal_im_detect(proposal_detection_model.conf_proposal, rpn_net, im, true);
    t_proposal = toc(th);
%     norm_feats = rpn_net.blobs('normalized_features').get_data();
%     norm_feats = permute(norm_feats, [2, 1, 3, 4]);
%     for i=1:size(norm_feats,3)
%         imshow(norm_feats(:,:,i),[])
%         title(strcat(num2str(i), ' normalized features'));
%         pause
%     end
%     pool_feats = rpn_net.blobs('pool1').get_data();
%     pool_feats = permute(pool_feats, [2, 1, 3, 4]);
%     for i=1:size(pool_feats,3)
%         imshow(pool_feats(:,:,i),[])
%         title(strcat(num2str(i), ' pooled features'));
%         pause
%     end
%     conv_feats = rpn_net.blobs('conv1').get_data();
%     for i=1:size(conv_feats,3)
%         imshow(conv_feats(:,:,i),[])
%         title(num2str(i));
%         pause
%     end
    th = tic();
    aboxes                      = boxes_filter([boxes, scores], opts.per_nms_topN, opts.nms_overlap_thres, opts.after_nms_topN, opts.use_gpu);
    t_nms = toc(th);
    
    % test detection
    th = tic();    
    [boxes, scores]             = fast_rcnn_im_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
                                  aboxes(:, 1:4), opts.after_nms_topN, true);
%     conv_feats = fast_rcnn_net.blobs('conv1').get_data();
%     for i=1:size(conv_feats,3)
%         imshow(conv_feats(:,:,i),[])
%         title(strcat(num2str(i), 'fast rcnn'));
%         pause
%     end
    
    t_detection = toc(th);
    
    fprintf('%s (%dx%d): time %.3fs (resize+conv+proposal: %.3fs, nms+regionwise: %.3fs)\n', im_names(j).name, ...
        size(im, 2), size(im, 1), t_proposal + t_nms + t_detection, t_proposal, t_nms+t_detection);
    running_time(end+1) = t_proposal + t_nms + t_detection;
    
    % visualize
    classes = proposal_detection_model.classes;
    boxes_cell = cell(length(classes), 1);
    thres = 0.01;
    for i = 1:length(boxes_cell)
        boxes_cell{i} = [boxes(:, (1+(i-1)*4):(i*4)), scores(:, i)];
        boxes_cell{i} = boxes_cell{i}(nms(boxes_cell{i}, 0.3), :);
        
        I = boxes_cell{i}(:, 5) >= thres;
        boxes_cell{i} = boxes_cell{i}(I, :);
    end
    figure(j);
    %showboxes(imread(fullfile(pwd, im_names{j})), boxes_cell, classes, 'voc');
    showboxes(imread(fullfile(im_dir, im_names(j).name)), boxes_cell, classes, 'voc');
    pause(0.1);
end
fprintf('mean time: %.3fs\n', mean(running_time));

caffe.reset_all(); 
clear mex;

end

function proposal_detection_model = load_proposal_detection_model(model_dir)   
    
    ld                          = load(fullfile(model_dir, 'model'));
    proposal_detection_model    = ld.proposal_detection_model;
    clear ld;
    
    proposal_detection_model.proposal_net_def ...
                                = fullfile(model_dir, 'rpn_test.prototxt');
    proposal_detection_model.proposal_net ...
                                = fullfile(model_dir, 'final_rpn');
    proposal_detection_model.detection_net_def ...
                                = fullfile(model_dir, 'fast_rcnn_test.prototxt');
    proposal_detection_model.detection_net ...
                                = fullfile(model_dir, 'final_fast_rcnn');
    
end

function aboxes = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
    % to speed up nms
    if per_nms_topN > 0
        aboxes = aboxes(1:min(length(aboxes), per_nms_topN), :);
    end
    % do nms
    if nms_overlap_thres > 0 && nms_overlap_thres < 1
        aboxes = aboxes(nms(aboxes, nms_overlap_thres, use_gpu), :);       
    end
    if after_nms_topN > 0
        aboxes = aboxes(1:min(length(aboxes), after_nms_topN), :);
    end
end
